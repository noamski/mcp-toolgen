"""Basic tests for mcp_toolgen package."""

import pytest
from mcp_toolgen import generate_tools_from_graphql


def test_import():
    """Test that the main functions can be imported."""
    assert callable(generate_tools_from_graphql)


def test_generate_tools_from_graphql_with_empty_schema():
    """Test generate_tools_from_graphql with minimal schema."""
    # Minimal valid GraphQL introspection result
    minimal_schema = {
        "data": {
            "__schema": {
                "types": [
                    {
                        "kind": "OBJECT",
                        "name": "Query",
                        "fields": []
                    },
                    {
                        "kind": "OBJECT", 
                        "name": "Mutation",
                        "fields": []
                    }
                ]
            }
        }
    }
    
    result = generate_tools_from_graphql(minimal_schema)
    assert isinstance(result, list)
    assert len(result) == 0  # No fields means no tools


def test_generate_tools_invalid_format():
    """Test that invalid format raises ValueError."""
    minimal_schema = {
        "data": {
            "__schema": {
                "types": []
            }
        }
    }
    
    with pytest.raises(ValueError, match="fmt must be 'openai' or 'claude'"):
        generate_tools_from_graphql(minimal_schema, fmt="invalid")


def test_generate_tools_claude_format_with_descriptions():
    """Test GraphQL to Claude JSON format with proper descriptions for queries and mutations."""
    # Complete GraphQL schema with both Query and Mutation types
    graphql_schema = {
        "data": {
            "__schema": {
                "types": [
                    {
                        "kind": "OBJECT",
                        "name": "Query",
                        "description": "Root query type",
                        "fields": [
                            {
                                "name": "getUser",
                                "description": "Retrieve a user by their unique identifier",
                                "args": [
                                    {
                                        "name": "id",
                                        "description": "The unique identifier of the user",
                                        "type": {
                                            "kind": "NON_NULL",
                                            "ofType": {
                                                "kind": "SCALAR",
                                                "name": "ID"
                                            }
                                        }
                                    }
                                ]
                            },
                            {
                                "name": "searchUsers",
                                "description": "Search for users by name or email",
                                "args": [
                                    {
                                        "name": "query",
                                        "description": "Search query string",
                                        "type": {
                                            "kind": "SCALAR",
                                            "name": "String"
                                        }
                                    },
                                    {
                                        "name": "limit",
                                        "description": "Maximum number of results to return",
                                        "type": {
                                            "kind": "SCALAR",
                                            "name": "Int"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "kind": "OBJECT",
                        "name": "Mutation",
                        "description": "Root mutation type",
                        "fields": [
                            {
                                "name": "createUser",
                                "description": "Create a new user account with the provided information",
                                "args": [
                                    {
                                        "name": "input",
                                        "description": "User creation input data",
                                        "type": {
                                            "kind": "NON_NULL",
                                            "ofType": {
                                                "kind": "INPUT_OBJECT",
                                                "name": "CreateUserInput"
                                            }
                                        }
                                    }
                                ]
                            },
                            {
                                "name": "updateUser",
                                "description": "Update an existing user's information",
                                "args": [
                                    {
                                        "name": "id",
                                        "description": "The ID of the user to update",
                                        "type": {
                                            "kind": "NON_NULL",
                                            "ofType": {
                                                "kind": "SCALAR",
                                                "name": "ID"
                                            }
                                        }
                                    },
                                    {
                                        "name": "input",
                                        "description": "Updated user data",
                                        "type": {
                                            "kind": "INPUT_OBJECT",
                                            "name": "UpdateUserInput"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "kind": "INPUT_OBJECT",
                        "name": "CreateUserInput",
                        "description": "Input type for creating a new user",
                        "inputFields": [
                            {
                                "name": "name",
                                "description": "The user's full name",
                                "type": {
                                    "kind": "NON_NULL",
                                    "ofType": {
                                        "kind": "SCALAR",
                                        "name": "String"
                                    }
                                }
                            },
                            {
                                "name": "email",
                                "description": "The user's email address",
                                "type": {
                                    "kind": "NON_NULL",
                                    "ofType": {
                                        "kind": "SCALAR",
                                        "name": "String"
                                    }
                                }
                            },
                            {
                                "name": "age",
                                "description": "The user's age in years",
                                "type": {
                                    "kind": "SCALAR",
                                    "name": "Int"
                                }
                            }
                        ]
                    },
                    {
                        "kind": "INPUT_OBJECT",
                        "name": "UpdateUserInput",
                        "description": "Input type for updating user information",
                        "inputFields": [
                            {
                                "name": "name",
                                "description": "Updated user name",
                                "type": {
                                    "kind": "SCALAR",
                                    "name": "String"
                                }
                            },
                            {
                                "name": "email",
                                "description": "Updated email address",
                                "type": {
                                    "kind": "SCALAR",
                                    "name": "String"
                                }
                            }
                        ]
                    }
                ]
            }
        }
    }
    
    # Generate tools in Claude format
    result = generate_tools_from_graphql(graphql_schema, fmt="claude")
    
    # Verify we got the expected number of tools (2 queries + 2 mutations)
    assert len(result) == 4
    
    # Find each tool by name
    tools_by_name = {tool["name"]: tool for tool in result}
    
    # Test Query: getUser
    get_user = tools_by_name["getUser"]
    assert get_user["name"] == "getUser"
    assert get_user["description"] == "Retrieve a user by their unique identifier"
    assert "input_schema" in get_user  # Claude format uses input_schema
    assert get_user["input_schema"]["type"] == "object"
    assert "id" in get_user["input_schema"]["properties"]
    assert get_user["input_schema"]["properties"]["id"]["type"] == "string"
    assert get_user["input_schema"]["properties"]["id"]["description"] == "The unique identifier of the user"
    assert get_user["input_schema"]["required"] == ["id"]
    
    # Test Query: searchUsers
    search_users = tools_by_name["searchUsers"]
    assert search_users["name"] == "searchUsers"
    assert search_users["description"] == "Search for users by name or email"
    assert "query" in search_users["input_schema"]["properties"]
    assert "limit" in search_users["input_schema"]["properties"]
    assert search_users["input_schema"]["properties"]["query"]["type"] == "string"
    assert search_users["input_schema"]["properties"]["limit"]["type"] == "integer"
    assert search_users["input_schema"]["properties"]["query"]["description"] == "Search query string"
    assert search_users["input_schema"]["properties"]["limit"]["description"] == "Maximum number of results to return"
    assert "required" not in search_users["input_schema"]  # No required fields
    
    # Test Mutation: createUser
    create_user = tools_by_name["createUser"]
    assert create_user["name"] == "createUser"
    assert create_user["description"] == "Create a new user account with the provided information"
    assert "input" in create_user["input_schema"]["properties"]
    assert create_user["input_schema"]["properties"]["input"]["type"] == "object"
    assert create_user["input_schema"]["properties"]["input"]["description"] == "User creation input data"
    assert create_user["input_schema"]["required"] == ["input"]
    
    # Test nested input object properties
    input_props = create_user["input_schema"]["properties"]["input"]["properties"]
    assert "name" in input_props
    assert "email" in input_props
    assert "age" in input_props
    assert input_props["name"]["type"] == "string"
    assert input_props["email"]["type"] == "string"
    assert input_props["age"]["type"] == "integer"
    assert input_props["name"]["description"] == "The user's full name"
    assert input_props["email"]["description"] == "The user's email address"
    assert input_props["age"]["description"] == "The user's age in years"
    assert create_user["input_schema"]["properties"]["input"]["required"] == ["name", "email"]
    
    # Test Mutation: updateUser
    update_user = tools_by_name["updateUser"]
    assert update_user["name"] == "updateUser"
    assert update_user["description"] == "Update an existing user's information"
    assert "id" in update_user["input_schema"]["properties"]
    assert "input" in update_user["input_schema"]["properties"]
    assert update_user["input_schema"]["properties"]["id"]["type"] == "string"
    assert update_user["input_schema"]["properties"]["id"]["description"] == "The ID of the user to update"
    assert update_user["input_schema"]["properties"]["input"]["description"] == "Updated user data"
    assert update_user["input_schema"]["required"] == ["id"]
    
    # Verify the structure is valid Claude format
    for tool in result:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool  # Claude format
        assert "parameters" not in tool  # OpenAI format should not be present
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"] 