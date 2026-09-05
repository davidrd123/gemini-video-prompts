import json
from types import SimpleNamespace

import httpx
import pytest
from google import genai
from google.genai import types
from PIL import Image

from media_analysis_mcp import gemini_media, server


@pytest.mark.parametrize('matched', [True, False])
def test_agentic_sdk_wire_request_and_trace(matched):
    def respond(request):
        body = json.loads(request.content)
        assert body['store'] is False
        assert body['generation_config'] == {'thinking_level': 'high', 'max_output_tokens': 12000}
        body['input'] = body['input'][0]['content']
        assert [p['type'] for p in body['input']] == ['text', 'image', 'video', 'video']
        assert all(p['processing'] == 'agentic' for p in body['input'][2:])
        assert body['input'][1]['data']
        return httpx.Response(200, json={
            'id': 'test', 'status': 'completed', 'steps': [
                {'type': 'processing_call', 'id': 'c1'},
                {'type': 'processing_result', 'call_id': 'c1' if matched else 'other'},
                {'type': 'model_output', 'content': [{'type': 'text', 'text': 'Cut at 14s.'}]},
            ],
        })
    client = genai.Client(api_key='fake', http_options=types.HttpOptions(
        client_args={'transport': httpx.MockTransport(respond)}))
    result = gemini_media.call_agentic_video(
        client=client, model='gemini-3.8-flash', system_instruction='Inspect.',
        contents=['Question', Image.new('RGB', (2, 2)), *[
            types.Part(file_data=types.FileData(file_uri=f'https://example.com/{i}', mime_type='video/mp4'))
            for i in range(2)]],
        temperature=None, thinking_level='high', max_output_tokens=12000)
    assert result['answer'] == 'Cut at 14s.'
    assert result['agentic_processing_verified'] is matched
    assert len(result['processing_trace']) == 2
    assert result['interaction']['id'] == 'test'


@pytest.mark.parametrize('multiple', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_tool_dispatch_and_cleanup(tmp_path, monkeypatch, multiple, fail):
    paths = [tmp_path / f'{i}.mp4' for i in range(2)]
    for p in paths:
        p.touch()
    cleaned = []
    monkeypatch.setattr(gemini_media, 'init_client', lambda: (object(), types))
    monkeypatch.setattr(gemini_media, 'upload_and_poll_video', lambda c, p, **kw: SimpleNamespace(uri=p))
    monkeypatch.setattr(gemini_media, 'cleanup_uploaded', lambda c, p: cleaned.append(p.uri))
    def call(**kwargs):
        assert sum(getattr(p, 'file_data', None) is not None for p in kwargs['contents']) == (2 if multiple else 1)
        if fail:
            raise RuntimeError('failed')
        return {'answer': 'ok', 'processing_trace': [], 'agentic_processing_verified': False}
    monkeypatch.setattr(gemini_media, 'call_agentic_video', call)
    tool = server.analyze_videos if multiple else server.analyze_video
    args = {'video_paths': list(map(str, paths))} if multiple else {'video_path': str(paths[0])}
    if fail:
        with pytest.raises(RuntimeError, match='failed'):
            tool(**args, question='Where?', processing='agentic')
    else:
        result = tool(**args, question='Where?', processing='agentic')
        assert result['processing'] == 'agentic'
        assert result['answer'] == 'ok'
    assert len(cleaned) == (2 if multiple else 1)


@pytest.mark.parametrize('processing,fps', [('agentic', 1), ('invalid', None)])
def test_invalid_processing_before_client(tmp_path, monkeypatch, processing, fps):
    path = tmp_path / 'v.mp4'
    path.touch()
    monkeypatch.setattr(gemini_media, 'init_client', lambda: pytest.fail('must validate before API'))
    with pytest.raises(RuntimeError, match='INVALID_INPUT'):
        server.analyze_video(str(path), 'Where?', processing=processing, fps=fps)
