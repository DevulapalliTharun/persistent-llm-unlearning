"""Private adapter/optimizer snapshot backup with immutable hash round-trip receipts."""

from pathlib import Path
import tempfile
import time

from src.m0_knowledge_audit import sha256_file


def artifact_hashes(directory):
    directory = Path(directory)
    paths = [p for folder in ('adapter', 'tokenizer') for p in (directory / folder).rglob('*') if p.is_file()]
    paths += [directory / 'optimizer_state.pt']
    if (not (directory / 'adapter/adapter_model.safetensors').is_file()
            or not (directory / 'tokenizer/tokenizer_config.json').is_file()
            or not all(p.is_file() for p in paths)):
        raise ValueError('Snapshot requires a saved adapter, tokenizer and optimizer state')
    return {str(p.relative_to(directory)): sha256_file(p) for p in sorted(paths)}


def publish_verified(directory, settings, remote_path, token, api=None, downloader=None):
    """Back up only to the declared private repo. Never print or serialize a token."""
    directory = Path(directory)
    if not token or not settings['require_private'] or not settings['checkpoint_roundtrip_required']:
        raise ValueError('Private verified backup is required')
    if not remote_path.startswith(settings['path_prefix'] + '/'):
        raise ValueError('Backup path is outside the declared full-M1 prefix')
    if api is None or downloader is None:
        from huggingface_hub import HfApi, hf_hub_download
        api = api or HfApi(token=token)
        downloader = downloader or hf_hub_download
    if not api.repo_info(settings['repo_id'], repo_type='model').private:
        raise ValueError('Artifact repository must be private')
    import json
    manifest = json.loads((directory / 'artifact_manifest.json').read_text())
    expected = manifest['artifact_files_sha256']
    if artifact_hashes(directory) != expected:
        raise ValueError('Saved snapshot bytes differ from manifest')
    started = time.perf_counter()
    commit = api.upload_folder(repo_id=settings['repo_id'], repo_type='model', folder_path=str(directory),
                              path_in_repo=remote_path, commit_message='Preserve full Qwen3-1.7B M1 candidate evidence')
    with tempfile.TemporaryDirectory(prefix='m1-full-roundtrip-') as temp:
        for name, digest in expected.items():
            restored = downloader(repo_id=settings['repo_id'], filename=remote_path + '/' + name,
                                  revision=commit.oid, token=token, local_dir=temp)
            if sha256_file(Path(restored)) != digest:
                raise ValueError(f'Private snapshot round-trip mismatch: {name}')
    receipt = {'repo_id': settings['repo_id'], 'artifact_revision': commit.oid, 'path': remote_path,
            'artifact_files_sha256': expected, 'all_artifact_files_verified': True,
            'canonical_m1': False, 'seconds': round(time.perf_counter() - started, 4)}
    from src.m0_smoke_test import write_json
    write_json(directory / 'persistence.json', receipt)
    api.upload_file(repo_id=settings['repo_id'], repo_type='model',
                    path_or_fileobj=str(directory / 'persistence.json'), path_in_repo=remote_path + '/persistence.json',
                    commit_message='Record verified full-M1 snapshot round-trip')
    return receipt
