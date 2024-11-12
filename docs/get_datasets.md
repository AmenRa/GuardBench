# Get the datasets

> [!WARNING]  
> By downloading and using the datasets provided by GuardBench, you agree with the license of the specific datasets you intend to use. If you are unsure about what you can and can't do with specific datasets, please refer to their original repositories and owners.

## HuggingFace CLI Login
To download some datasets you need to be logged into [HuggingFace](https://huggingface.co).
If you don't have an account, [create one](https://huggingface.co/join).
Then, create/copy an access token from your [profile settings](https://huggingface.co/settings/tokens).
Lastly, log into HuggingFace using the [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```sh
huggingface-cli login
```
## DecodingTrust
To download DecodingTrust, you must first fill [this form](https://huggingface.co/datasets/AI-Secure/DecodingTrust).

## HEx-PHI
To download HEx-PHI, you must first fill [this form](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI).

## ToxiGen
To download ToxiGen, you must first fill [this form](https://forms.office.com/pages/responsepage.aspx?id=utmvZM8Oz0q8NpNfYjW6i0NV-wEU-P5EvmfM4BAGgCJUNThXTDdMT1hXSjhOQ1pGRUtXVEJMSTVVRy4u).
We suggest to use the same email address of your HuggingFace account.

## Download Datasets
Datasets are downloaded and formatted on the first request and stored in `~/.guardbench/datasets` for later use.  
You can also download and format all datasets in advance with the following command:

```bash
python -c "import guardbench; guardbench.download_all()"
```