"""Print 'name<TAB>url' for every output file of a Kaggle kernel (for resumable downloads with curl)."""
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

api = KaggleApi()
api.authenticate()
owner, slug = sys.argv[1].split("/")
token = None
with api.build_kaggle_client() as kaggle:
    while True:
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug, req.page_size = owner, slug, 100
        if token:
            req.page_token = token
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        for f in resp.files:
            print(f"{f.file_name}\t{f.url}")
        token = resp.next_page_token
        if not token:
            break
