import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict


def _generate_id(prefix: str = "resp_") -> str:
    return prefix + uuid.uuid4().hex


def translate_request_to_chat_completions(req_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translates a Responses API request body into a Chat Completions request body.
    """
    chat_req = {}
    
    # Pass through standard fields
    for field in ["model", "temperature", "top_p", "tools", "tool_choice", "stream"]:
        if field in req_body:
            chat_req[field] = req_body[field]
            
    # Map max_output_tokens -> max_completion_tokens
    if "max_output_tokens" in req_body:
        chat_req["max_completion_tokens"] = req_body["max_output_tokens"]

    messages = []
    
    # Map instructions -> system message
    if "instructions" in req_body and req_body["instructions"]:
        messages.append({
            "role": "system",
            "content": req_body["instructions"]
        })
        
    # Map input -> user messages
    if "input" in req_body:
        input_data = req_body["input"]
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            # The Responses API `input` can be an array of message objects or content parts
            for item in input_data:
                if isinstance(item, dict) and "role" in item:
                    # Message object format in input array
                    msg = {"role": item["role"]}
                    content = item.get("content", "")
                    if isinstance(content, list):
                        chat_content = []
                        for part in content:
                            if part.get("type") == "input_text":
                                chat_content.append({"type": "text", "text": part.get("text", "")})
                            elif part.get("type") == "input_image":
                                chat_content.append({"type": "image_url", "image_url": {"url": part.get("image_url", "")}})
                            else:
                                chat_content.append(part)
                        msg["content"] = chat_content
                    else:
                        msg["content"] = content
                    messages.append(msg)
                else:
                    # In case it's a flat array of strings or unknown
                    messages.append({"role": "user", "content": str(item)})
                    
    chat_req["messages"] = messages
    return chat_req


def translate_response_to_responses_api(chat_resp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translates a Chat Completions JSON response back to a Responses API payload.
    """
    resp_id = _generate_id("resp_")
    msg_id = _generate_id("msg_")
    created_at = chat_resp.get("created", int(time.time()))
    model = chat_resp.get("model", "unknown")
    
    output = []
    for choice in chat_resp.get("choices", []):
        message = choice.get("message", {})
        
        # Responses API outputs are nested
        content_parts = []
        if "content" in message and message["content"]:
            content_parts.append({
                "type": "output_text",
                "text": message["content"],
                "annotations": []
            })
            
        output_item = {
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": message.get("role", "assistant"),
            "content": content_parts
        }
        
        if "tool_calls" in message:
            # Map tool calls
            for tc in message["tool_calls"]:
                output.append({
                    "type": "function_call",
                    "id": tc.get("id"),
                    "call_id": tc.get("id"),
                    "name": tc.get("function", {}).get("name"),
                    "arguments": tc.get("function", {}).get("arguments"),
                    "status": "completed"
                })
        
        if content_parts:
            output.append(output_item)
            
    usage = chat_resp.get("usage", {})
    
    return {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "completed_at": created_at + 1,
        "error": None,
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
    }


async def translate_stream_to_responses_api(lines_generator, model_name: str, req_body: dict) -> AsyncGenerator[str, None]:
    """
    Consumes SSE lines from a Chat Completions stream and yields Responses API SSE events.
    """
    resp_id = _generate_id("resp_")
    msg_id = _generate_id("msg_")
    created_at = int(time.time())
    
    instructions = req_body.get("instructions", None)
    
    def _build_event(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    # 1. response.created
    yield _build_event("response.created", {
        "type": "response.created",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "instructions": instructions,
            "model": model_name,
            "output": []
        }
    })
    
    # 2. response.in_progress
    yield _build_event("response.in_progress", {
        "type": "response.in_progress",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "instructions": instructions,
            "model": model_name,
            "output": []
        }
    })
    
    # 3. response.output_item.added
    yield _build_event("response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "id": msg_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": []
        }
    })

    # 4. response.content_part.added
    yield _build_event("response.content_part.added", {
        "type": "response.content_part.added",
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "part": {
            "type": "output_text",
            "text": "",
            "annotations": []
        }
    })

    full_text = ""
    finish_reason = None
    final_usage = None
    
    async for line in lines_generator:
        line = line.decode("utf-8").strip() if isinstance(line, bytes) else line.strip()
        if not line or not line.startswith("data: "):
            continue
            
        data_str = line[6:]
        if data_str == "[DONE]":
            break
            
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
            
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            if "content" in delta and delta["content"]:
                text_delta = delta["content"]
                full_text += text_delta
                
                # Yield text delta
                yield _build_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text_delta
                })
                
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
                
        if "usage" in chunk and chunk["usage"]:
            final_usage = chunk["usage"]

    # End sequence
    
    # 5. response.output_text.done
    yield _build_event("response.output_text.done", {
        "type": "response.output_text.done",
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "text": full_text
    })
    
    # 6. response.content_part.done
    yield _build_event("response.content_part.done", {
        "type": "response.content_part.done",
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "part": {
            "type": "output_text",
            "text": full_text,
            "annotations": []
        }
    })
    
    # 7. response.output_item.done
    yield _build_event("response.output_item.done", {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": full_text,
                    "annotations": []
                }
            ]
        }
    })
    
    # 8. response.completed
    usage_dict = None
    if final_usage:
        usage_dict = {
            "input_tokens": final_usage.get("prompt_tokens", 0),
            "output_tokens": final_usage.get("completion_tokens", 0),
            "total_tokens": final_usage.get("total_tokens", 0)
        }
        
    yield _build_event("response.completed", {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "instructions": instructions,
            "model": model_name,
            "output": [
                {
                    "id": msg_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": full_text,
                            "annotations": []
                        }
                    ]
                }
            ],
            "usage": usage_dict
        }
    })
