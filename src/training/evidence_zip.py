"""Archive text evidence, including ignored checkpoint JSON, and privately verify the ZIP."""

import hashlib
import json
from pathlib import Path
import zipfile


def archive_evidence(directory):
    directory = Path(directory)
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix in {'.json', '.jsonl', '.txt', '.py'}]
    for checkpoint in (directory / 'checkpoints').glob('attempt-*'):
        files += [checkpoint / name for name in ('checkpoint.json', 'artifact_manifest.json', 'persistence.json')
                  if (checkpoint / name).is_file()]
    if not files:
        raise ValueError('No text evidence to export')
    files.sort()
    archive = Path(str(directory) + '-evidence.zip')
    if archive.exists():
        with zipfile.ZipFile(archive) as saved:
            expected = {str(p.relative_to(directory)) for p in files}
            if (saved.testzip() is not None or len(saved.infolist()) != len(expected)
                    or set(saved.namelist()) != expected
                    or any(saved.read(str(p.relative_to(directory))) != p.read_bytes() for p in files)):
                raise ValueError('Existing ZIP differs; preserve it rather than overwrite')
    else:
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as saved:
            for path in files:
                saved.write(path, arcname=str(path.relative_to(directory)))
    return archive


def publish_zip(archive, settings, path_prefix, token, api=None, downloader=None):
    if not token or not settings['require_private']:
        raise ValueError('Private authentication required for ZIP backup')
    if api is None or downloader is None:
        from huggingface_hub import HfApi, hf_hub_download
        api = api or HfApi(token=token)
        downloader = downloader or hf_hub_download
    if not api.repo_info(settings['repo_id'], repo_type='model').private:
        raise ValueError('Evidence repository must be private')
    archive = Path(archive)
    remote = path_prefix + '/' + archive.name.removesuffix('-evidence.zip') + '/' + archive.name
    commit = api.upload_file(repo_id=settings['repo_id'], repo_type='model', path_or_fileobj=str(archive),
                             path_in_repo=remote, commit_message='Preserve inverse-support M1 evidence ZIP')
    restored = Path(downloader(repo_id=settings['repo_id'], repo_type='model', filename=remote,
                               revision=commit.oid, token=token))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if hashlib.sha256(restored.read_bytes()).hexdigest() != digest:
        raise ValueError('Evidence ZIP round-trip mismatch')
    receipt = {'repo_id': settings['repo_id'], 'revision': commit.oid, 'path': remote, 'sha256': digest,
               'zip_roundtrip_verified': True}
    Path(str(archive) + '-persistence.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
