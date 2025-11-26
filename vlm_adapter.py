"""Example remote VLM adapter callback stub.

Place this file in the project and set the environment variable:
  VLM_ADAPTER_CALLBACK=vlm_adapter:remote_inference

The function `remote_inference(image_path, prompt)` will be imported and
called by `DentAdapt` when `USE_VLM_ADAPTER=1` and `VLM_ADAPTER_CALLBACK` is set.

This is a minimal example showing how to forward the image and prompt to a
remote multimodal API (HTTP). Update `VLM_API_URL` and authentication as
needed for your provider.
"""
import os
import requests

def remote_inference(image_path, prompt):
    """Send image + prompt to a remote VLM endpoint and return the textual reply.

    The remote endpoint is expected to accept multipart/form-data with fields:
      - 'image' : file
      - 'prompt' : string

    And return JSON with a top-level 'text' field containing the answer.
    """
    api_url = os.getenv('VLM_API_URL')
    api_key = os.getenv('VLM_API_KEY')

    if not api_url:
        raise RuntimeError('VLM_API_URL not set')

    headers = {}
    if api_key:
        # Common pattern: Bearer token
        headers['Authorization'] = f'Bearer {api_key}'

    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'application/octet-stream')}
        data = {'prompt': prompt}
        resp = requests.post(api_url, headers=headers, files=files, data=data, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f'VLM request failed: {resp.status_code} {resp.text}')

    try:
        j = resp.json()
        # provider-specific: try common keys
        if isinstance(j, dict):
            for k in ('text', 'answer', 'reply'):
                if k in j:
                    return j[k]
        # fallback: return full JSON string
        return str(j)
    except Exception:
        return resp.text
