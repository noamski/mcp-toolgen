"""mcp_toolgen
================
Tiny library **and** CLI to auto‑generate AI tool / function specs from:

* **GraphQL** – given either an introspection JSON *file* **or** a live
  endpoint URL (with optional HTTP headers such as `Authorization`).
* **gRPC** – given a compiled **descriptor set** (`*.desc`).

It outputs a list of functions compatible with **OpenAI** (`functions=`)
*or* **Anthropic Claude** (`tools=`) formats.

Quick demo – GraphQL endpoint with Bearer token::

    $ mcp_toolgen --url https://api.acme.com/graphql \
           --header "Authorization: Bearer $TOKEN" \
           --only-mutations --format openai > tools.json

Python use::

    import mcp_toolgen as tg, os, json
    tools = tg.generate_tools_from_graphql(
        url="https://api.acme.com/graphql",
        headers={"Authorization": f"Bearer {os.environ['TOKEN']}"},
        only_mutations=True,
    )
    openai.ChatCompletion.create(model="gpt-4o", messages=msgs, functions=tools)

Dependencies:  `requests`  `graphql-core`  `protobuf`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Union, cast

import requests  # HTTP fetch for live GraphQL introspection

try:
    from google.protobuf import descriptor_pb2
    from google.protobuf.descriptor_pb2 import FieldDescriptorProto as FDP
except ImportError:  # pragma: no cover
    descriptor_pb2 = None  # type: ignore
    FDP = None  # type: ignore

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "generate_tools_from_graphql",
    "generate_tools_from_proto",
]

# ---------------------------------------------------------------------------
# Helpers – GraphQL
# ---------------------------------------------------------------------------
INTROSPECTION_QUERY = """
query IntrospectionQuery {
  __schema {
    types {
      kind
      name
      description
      fields(includeDeprecated: true) {
        name
        description
        args {
          name
          description
          type { ...TypeRef }
        }
      }
      inputFields {
        name
        description
        type { ...TypeRef }
      }
      enumValues(includeDeprecated: true) { name description }
    }
  }
}

fragment TypeRef on __Type {
  kind
  name
  ofType { kind name ofType { kind name ofType { kind name } } }
}
"""

_SCALAR_MAP = {
    "Int": {"type": "integer"},
    "Float": {"type": "number"},
    "String": {"type": "string"},
    "Boolean": {"type": "boolean"},
    "ID": {"type": "string"},
}


def _fetch_introspection(url: str, headers: Optional[MutableMapping[str, str]] = None) -> Dict[str, Any]:
    resp = requests.post(url, json={"query": INTROSPECTION_QUERY}, headers=headers or {})
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data:
        raise RuntimeError(f"Introspection failed: {data}")
    return data


def _load_introspection(source: Union[str, pathlib.Path, Dict[str, Any]],
                        headers: Optional[MutableMapping[str, str]] = None) -> Dict[str, Any]:
    if isinstance(source, (str, pathlib.Path)) and str(source).startswith(("http://", "https://")):
        return _fetch_introspection(str(source), headers)
    if isinstance(source, (str, pathlib.Path)):
        with open(source, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return cast(Dict[str, Any], source)


def _gql_type_index(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {t["name"]: t for t in schema["data"]["__schema"]["types"]}


def _is_nonnull(t: Dict[str, Any]) -> bool:
    return t.get("kind") == "NON_NULL"


def _to_schema(t: Dict[str, Any], idx: Dict[str, Any]) -> Dict[str, Any]:
    kind = t.get("kind")
    if kind == "NON_NULL":
        return _to_schema(t["ofType"], idx)
    if kind == "LIST":
        return {"type": "array", "items": _to_schema(t["ofType"], idx)}
    if kind == "SCALAR":
        return dict(_SCALAR_MAP.get(t["name"], {"type": "string"}))
    if kind == "ENUM":
        vals = [v["name"] for v in idx[t["name"]]["enumValues"]]
        return {"type": "string", "enum": vals}
    if kind == "INPUT_OBJECT":
        props, req = {}, []
        for f in idx[t["name"]]["inputFields"]:
            props[f["name"]] = _to_schema(f["type"], idx)
            if desc := f.get("description"):
                props[f["name"]]["description"] = desc
            if _is_nonnull(f["type"]):
                req.append(f["name"])
        obj: Dict[str, Any] = {"type": "object", "properties": props}
        if req:
            obj["required"] = req
        return obj
    return {"type": "string"}


def _build_fn(field: Dict[str, Any], idx: Dict[str, Any], fmt: str) -> Dict[str, Any]:
    props, req = {}, []
    for arg in field.get("args", []):
        props[arg["name"]] = _to_schema(arg["type"], idx)
        if d := arg.get("description"):
            props[arg["name"]]["description"] = d
        if _is_nonnull(arg["type"]):
            req.append(arg["name"])
    params: Dict[str, Any] = {"type": "object", "properties": props}
    if req:
        params["required"] = req
    key = "parameters" if fmt == "openai" else "input_schema"
    return {
        "name": field["name"],
        "description": (field.get("description") or f"Executes {field['name']}").strip(),
        key: params,
    }


def generate_tools_from_graphql(
    source: Union[str, pathlib.Path, Dict[str, Any]],
    *,
    headers: Optional[MutableMapping[str, str]] = None,
    only_mutations: bool = False,
    fmt: str = "openai",
) -> List[Dict[str, Any]]:
    """Return tool specs from GraphQL (file *or* live endpoint).

    `source` can be:
      * path to introspection JSON file
      * full URL of GraphQL endpoint (http/https) – will fetch introspection
      * already-parsed dict
    """
    if fmt not in ("openai", "claude"):
        raise ValueError("fmt must be 'openai' or 'claude'")

    schema = _load_introspection(source, headers)
    idx = _gql_type_index(schema)

    fns: List[Dict[str, Any]] = []
    for root_name in ("Mutation",) if only_mutations else ("Mutation", "Query"):
        root = idx.get(root_name)
        if not root:
            continue
        for field in root.get("fields", []):
            fns.append(_build_fn(field, idx, fmt))
    return fns

# ---------------------------------------------------------------------------
# Proto helpers
# ---------------------------------------------------------------------------

def _proto_scalar(field):  # type: ignore[no-any-unbound]
    mapping = {
        FDP.TYPE_BOOL: {"type": "boolean"},
        FDP.TYPE_STRING: {"type": "string"},
        FDP.TYPE_BYTES: {"type": "string"},
        FDP.TYPE_DOUBLE: {"type": "number"},
        FDP.TYPE_FLOAT: {"type": "number"},
        FDP.TYPE_INT32: {"type": "integer"},
        FDP.TYPE_INT64: {"type": "integer"},
        FDP.TYPE_UINT32: {"type": "integer"},
        FDP.TYPE_UINT64: {"type": "integer"},
        FDP.TYPE_SINT32: {"type": "integer"},
        FDP.TYPE_SINT64: {"type": "integer"},
    }
    return mapping.get(field.type, {"type": "string"})


def _msg_schema(msg, fd_index):  # type: ignore[no-any-unbound]
    props, req = {}, []
    for f in msg.field:
        name = f.json_name or f.name
        if f.type == FDP.TYPE_MESSAGE:
            schema = _msg_schema(fd_index[f.type_name.lstrip(".")], fd_index)
        elif f.type == FDP.TYPE_ENUM:
            vals = [v.name for v in fd_index[f.type_name.lstrip(".")].value]
            schema = {"type": "string", "enum": vals}
        else:
            schema = _proto_scalar(f)
        if f.label == FDP.LABEL_REPEATED:
            schema = {"type": "array", "items": schema}
        props[name] = schema
        if f.label == FDP.LABEL_REQUIRED:
            req.append(name)
    obj: Dict[str, Any] = {"type": "object", "properties": props}
    if req:
        obj["required"] = req
    return obj


def _rpc_fn_name(rpc: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", rpc).lower()


def generate_tools_from_proto(
    descriptor_file: Union[str, pathlib.Path],
    *,
    services: Optional[Sequence[str]] = None,
    fmt: str = "openai",
) -> List[Dict[str, Any]]:
    if descriptor_pb2 is None:
        raise RuntimeError("protobuf package required for proto support")
    if fmt not in ("openai", "claude"):
        raise ValueError("fmt must be 'openai' or 'claude'")

    desc_set = descriptor_pb2.FileDescriptorSet()
    with open(descriptor_file, "rb") as fh:
        desc_set.ParseFromString(fh.read())

    fd_index = {}
    for fd in desc_set.file:
        for m in fd.message_type:
            fd_index[f".{fd.package}.{m.name}"] = m
        for e in fd.enum_type:
            fd_index[f".{fd.package}.{e.name}"] = e

    key = "parameters" if fmt == "openai" else "input_schema"
    out: List[Dict[str, Any]] = []
    for fd in desc_set.file:
        for svc in fd.service:
            if services and svc.name not in services:
                continue
            for rpc in svc.method:
                fn_schema = _msg_schema(fd_index[rpc.input_type], fd_index)
                out.append(
                    {
                        "name": _rpc_fn_name(rpc.name),
                        "description": (rpc.leading_comments.strip() if rpc.leading_comments else f"Calls {rpc.name}"),
                        key: fn_schema,
                    }
                )
    return out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_headers(header_list: List[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for h in header_list:
        if ":" not in h:
            raise argparse.ArgumentTypeError("Header must be in 'Key: Value' format")
        k, v = h.split(":", 1)
        headers[k.strip()] = v.strip()
    return headers


def _cli(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(prog="mcp_toolgen", description="Generate AI tool specs from GraphQL or proto.")
    src_grp = ap.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("source", nargs="?", help="Path to introspection JSON or .desc file")
    src_grp.add_argument("--url", help="GraphQL endpoint URL – introspection fetched automatically")
    ap.add_argument("--header", action="append", default=[], help="HTTP header for --url (e.g. 'Authorization: Bearer X')")
    ap.add_argument("--format", choices=["openai", "claude"], default="openai")
    ap.add_argument("--only-mutations", action="store_true", help="GraphQL: include only Mutation fields")
    ap.add_argument("--services", help="Proto: comma-separated service names to include")
    ns = ap.parse_args(argv)

    if ns.url:
        tools = generate_tools_from_graphql(ns.url, headers=_parse_headers(ns.header), only_mutations=ns.only_mutations, fmt=ns.format)
    else:
        if not ns.source:
            ap.error("Provide a source file or --url")
        path = pathlib.Path(ns.source)
        if path.suffix == ".desc":
            svcs = ns.services.split(",") if ns.services else None
            tools = generate_tools_from_proto(path, services=svcs, fmt=ns.format)
        else:
            tools = generate_tools_from_graphql(path, only_mutations=ns.only_mutations, fmt=ns.format)

    json.dump(tools, sys.stdout, indent=2)
    sys.stdout.write("\n")

if __name__ == "__main__":
    _cli()
