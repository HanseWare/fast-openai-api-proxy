from fastapi import APIRouter, Request

from utils import handle_request, handle_subresource_request, process_completion_response

router = APIRouter()


@router.post("/responses")
async def create_response(request: Request):
    """
    Creates a model response using the Responses API.
    Supports text/image inputs, tools, streaming, and more.
    
    POST /v1/responses
    """
    response = await handle_request(request, "v1/responses")
    return process_completion_response(response)


@router.post("/responses/compact")
async def compact_response(request: Request):
    """
    Compacts prior response items and returns a compacted response object.

    POST /v1/responses/compact
    """
    response = await handle_request(request, "v1/responses/compact")
    return process_completion_response(response)


@router.post("/responses/input_tokens")
async def count_response_input_tokens(request: Request):
    """
    Returns input token counts for a Responses request payload.

    POST /v1/responses/input_tokens
    """
    response = await handle_request(request, "v1/responses/input_tokens")
    return process_completion_response(response)


@router.get("/responses/{response_id}")
async def retrieve_response(request: Request, response_id: str):
    """
    Retrieves a previously created response by ID.
    
    GET /v1/responses/{response_id}
    """
    response = await handle_subresource_request(
        request,
        api_path="v1/responses",
        target_suffix=f"/{response_id}",
        method="GET",
    )
    return process_completion_response(response)


@router.delete("/responses/{response_id}")
async def delete_response(request: Request, response_id: str):
    """
    Passes through deletion for a stored response when the target provider supports it.

    DELETE /v1/responses/{response_id}
    """
    response = await handle_subresource_request(
        request,
        api_path="v1/responses",
        target_suffix=f"/{response_id}",
        method="DELETE",
    )
    return process_completion_response(response)


@router.post("/responses/{response_id}/cancel")
async def cancel_response(request: Request, response_id: str):
    """
    Cancels an in-progress response.
    
    POST /v1/responses/{response_id}/cancel
    """
    response = await handle_subresource_request(
        request,
        api_path="v1/responses",
        target_suffix=f"/{response_id}/cancel",
        method="POST",
    )
    return process_completion_response(response)


@router.post("/responses/{response_id}/input_items")
async def add_response_input_items(request: Request, response_id: str):
    """
    Adds input items to an in-progress response.
    Allows continuing multi-turn conversations.
    
    POST /v1/responses/{response_id}/input_items
    """
    response = await handle_subresource_request(
        request,
        api_path="v1/responses",
        target_suffix=f"/{response_id}/input_items",
        method="POST",
        allow_stream=True,
    )
    return process_completion_response(response)






