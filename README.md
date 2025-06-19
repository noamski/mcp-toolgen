# mcp-toolgen

Generate function-calling JSON schemas for OpenAI / Anthropic from your
GraphQL API or gRPC descriptor.

```bash
pip install mcp-toolgen[grpc]       # `[grpc]` adds protobuf support
mcp_toolgen --url https://api.acme.com/graphql \
            --header "Authorization: Bearer $TOKEN" \
            --only-mutations --format openai > functions.json
