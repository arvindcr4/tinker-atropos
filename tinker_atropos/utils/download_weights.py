import tinker
import urllib.request

# Replace this with your weights path output at the end of the training run
# <unique_id> is the Training Run ID in Tinker console
TINKER_PATH = "tinker://<unique_id>/sampler_weights/final"

OUTPUT_FILENAME = "archive.tar"

sc = tinker.ServiceClient()
rc = sc.create_rest_client()
future = rc.get_checkpoint_archive_url_from_tinker_path(TINKER_PATH)
checkpoint_archive_url_response = future.result()

urllib.request.urlretrieve(checkpoint_archive_url_response.url, OUTPUT_FILENAME)
