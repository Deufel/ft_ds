# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic==0.68.0",
#     "anyio==4.10.0",
#     "apsw==3.50.4.0",
#     "duckdb==1.4.0",
#     "fastcore==1.8.8",
#     "fastmigrate==0.4.0",
#     "httpx==0.28.1",
#     "ipython==9.5.0",
#     "mohtml==0.1.11",
#     "moterm==0.1.0",
#     "pytest==8.4.2",
#     "python-dateutil==2.9.0.post0",
#     "python-fasthtml==0.12.27",
#     "rich==14.1.0",
#     "sqlglot==27.16.3",
#     "starlette==0.48.0",
#     "uvicorn==0.35.0",
# ]
# ///

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import json,uuid,inspect,types,signal,asyncio,threading,inspect,random,contextlib

    #from fastcore.utils import *
    #from fastcore.xml import *
    from fastcore.meta import use_kwargs_dict

    from types import UnionType, SimpleNamespace as ns, GenericAlias
    from typing import Optional, get_type_hints, get_args, get_origin, Union, Mapping, TypedDict, List, Any
    from datetime import datetime,date
    from dataclasses import dataclass,fields
    from collections import namedtuple
    from inspect import isfunction,ismethod,Parameter,get_annotations
    from functools import wraps, partialmethod, update_wrapper
    from http import cookies
    from urllib.parse import urlencode, parse_qs, quote, unquote
    from copy import copy,deepcopy
    from warnings import warn
    from dateutil import parser as dtparse
    from httpx import ASGITransport, AsyncClient
    from anyio import from_thread
    from uuid import uuid4, UUID
    from base64 import b85encode,b64encode

    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.sessions import SessionMiddleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.middleware.authentication import AuthenticationMiddleware
    from starlette.authentication import AuthCredentials, AuthenticationBackend, AuthenticationError, SimpleUser, requires
    from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    from starlette.responses import Response, HTMLResponse, FileResponse, JSONResponse as JSONResponseOrig, RedirectResponse, StreamingResponse
    from starlette.requests import Request, HTTPConnection, FormData
    from starlette.staticfiles import StaticFiles
    from starlette.exceptions import HTTPException
    from starlette._utils import is_async_callable
    from starlette.convertors import Convertor, StringConvertor, register_url_convertor, CONVERTOR_TYPES
    from starlette.routing import Route, Router, Mount, WebSocketRoute
    from starlette.exceptions import HTTPException,WebSocketException
    from starlette.endpoints import HTTPEndpoint,WebSocketEndpoint
    from starlette.config import Config
    from starlette.datastructures import CommaSeparatedStrings, Secret, UploadFile, URLPath
    from starlette.types import ASGIApp, Receive, Scope, Send
    from starlette.concurrency import run_in_threadpool
    from starlette.background import BackgroundTask, BackgroundTasks
    from starlette.websockets import WebSocketDisconnect, WebSocket
    from starlette.requests import Headers

    # Global imports (not great for marimo but for dev this is (ithink) easiest way to import "chain import..")
    import importlib
    def import_all(*module_specs):
        for spec in module_specs:
            module_name, fallback = (spec, None) if isinstance(spec, str) else spec
            module = importlib.import_module(module_name)
            names = getattr(module, '__all__', fallback or [n for n in dir(module) if not n.startswith('_')])
            for name in names: globals()[name] = getattr(module, name)

    import_all('fastcore.utils', 'fastcore.xml')
    return (
        ASGITransport,
        Any,
        AsyncClient,
        BackgroundTask,
        BackgroundTasks,
        CORSMiddleware,
        FileResponse,
        FormData,
        GenericAlias,
        HTMLResponse,
        HTTPConnection,
        HTTPException,
        JSONResponseOrig,
        List,
        Mapping,
        Middleware,
        Optional,
        Parameter,
        RedirectResponse,
        Request,
        Response,
        Route,
        SessionMiddleware,
        Starlette,
        StreamingResponse,
        StringConvertor,
        URLPath,
        UUID,
        Union,
        UnionType,
        UploadFile,
        WebSocket,
        WebSocketEndpoint,
        WebSocketRoute,
        asyncio,
        b64encode,
        contextlib,
        cookies,
        dataclass,
        date,
        datetime,
        deepcopy,
        dtparse,
        from_thread,
        get_annotations,
        get_args,
        get_origin,
        inspect,
        is_async_callable,
        json,
        mo,
        ns,
        parse_qs,
        partialmethod,
        quote,
        random,
        register_url_convertor,
        run_in_threadpool,
        types,
        unquote,
        update_wrapper,
        urlencode,
        use_kwargs_dict,
        uuid,
        uuid4,
        warn,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Development Packages""")
    return


@app.cell
def _():
    from starlette.testclient import TestClient
    return (TestClient,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## FastHtml Core""")
    return


@app.cell(hide_code=True)
def _(mo):
    # Basic accordion with markdown content
    basic_accordion = mo.accordion({
        "Removed underscore prefixes from all internal functions:": """
    _params → get_function_params\n
    _get_htmx → get_htmx_headers\n
    _mk_list → make_list\n
    _fix_anno → fix_annotation\n
    _form_arg → form_arg\n
    _to_htmx_header → to_htmx_header\n
    _annotations → get_annotations_safe\n
    _is_body → is_body_type\n
    _formitem → get_form_item\n
    _from_body → from_body\n
    _find_p → find_param\n
    _wrap_req → wrap_request\n
    _handle → handle_async\n
    _find_wsp → find_websocket_param\n
    _wrap_ws → wrap_websocket\n
    _send_ws → send_websocket\n
    _ws_endp → websocket_endpoint\n
    _verbs → verbs\n
    _url_for → url_for\n
    _find_targets → find_targets\n
    _apply_ft → apply_ft\n
    _to_xml → to_xml_with_targets\n
    _iter_typs → iter_types\n
    _part_resp → partition_response\n
    _xt_cts → extract_content\n
    _is_ft_resp → is_ft_response\n
    _resp → create_response\n
    _wrap_call → wrap_function_call\n
    _list → make_list_safe\n
    _wrap_ex → wrap_exception_handler\n
    _endp → create_endpoint\n
    _add_ws → add_websocket_route\n
    _mk_locfunc → make_location_function\n
    _add_route → add_http_route\n
    _static_exts → static_extensions\n
    _add_ids → add_ids\n
        """,
        "Updated class attributes and method names:": """
    RouteFuncs._funcs → RouteFuncs.route_functions\n
    APIRouter._wrap_func → APIRouter.wrap_route_function\n
    Client._sync → Client.sync_request\n
    app._send → app.websocket_send\n
    MiddlewareBase._app → MiddlewareBase.app\n
        """,
        "Benifits For Marimo": """
        No more private function conflicts: Functions without underscores are properly accessible in marimo notebooks
    Better discoverability: All functions are now properly exposed and can be imported/used directly
    Cleaner namespace: The module follows Python conventions better without excessive use of "private" functions
    Improved debugging: Stack traces and error messages will show more readable function names

    The functionality remains exactly the same - only the naming has changed to be more marimo-friendly. You can now use this refactored version in your marimo notebooks without any accessibility issues.
        """ 
    })


    mo.vstack([
        basic_accordion,
    ])
    return


@app.cell
def _(
    ASGITransport,
    Any,
    AsyncClient,
    AttrDict,
    BackgroundTask,
    BackgroundTasks,
    Body,
    CORSMiddleware,
    FT,
    FileResponse,
    FormData,
    GenericAlias,
    HTMLResponse,
    HTTPConnection,
    HTTPException,
    Head,
    Html,
    IN_NOTEBOOK,
    JSONResponseOrig,
    Link,
    List,
    Mapping,
    Meta,
    Middleware,
    NotStr,
    Parameter,
    Path,
    RedirectResponse,
    Request,
    Response,
    Route,
    Script,
    SessionMiddleware,
    Starlette,
    StreamingResponse,
    StringConvertor,
    Title,
    URLPath,
    UUID,
    Union,
    UnionType,
    UploadFile,
    WebSocket,
    WebSocketEndpoint,
    WebSocketRoute,
    asyncio,
    b64encode,
    build_sse_event,
    camel2words,
    contextlib,
    cookies,
    dataclass,
    date,
    datetime,
    deepcopy,
    dict2obj,
    dtparse,
    first,
    format_datetime,
    from_thread,
    get_annotations,
    get_args,
    get_class,
    get_origin,
    ifnone,
    inspect,
    is_async_callable,
    is_namedtuple,
    json,
    listify,
    loads,
    noop,
    ns,
    os,
    parse_qs,
    partial,
    partialmethod,
    partition,
    patch,
    quote,
    random,
    re,
    register_url_convertor,
    risinstance,
    run_in_threadpool,
    signature_ex,
    snake2camel,
    str2bool,
    str2date,
    str2int,
    to_xml,
    tuplify,
    types,
    unquote,
    update_wrapper,
    urlencode,
    use_kwargs_dict,
    uuid,
    uuid4,
    warn,
):
    """The `FastHTML` subclass of `Starlette`, along with the `RouterX` and `RouteX` classes it automatically uses."""

    # AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/api/00_core.ipynb.

    # %% auto 0
    __all__ = ['empty', 'htmx_hdrs', 'fh_cfg', 'htmx_resps', 'htmx_exts', 'htmxsrc', 'fhjsscr', 'surrsrc', 'scopesrc', 'viewport',
               'charset', 'cors_allow', 'iframe_scr', 'all_meths', 'devtools_loc', 'parsed_date', 'snake2hyphens',
               'HtmxHeaders', 'HttpHeader', 'HtmxResponseHeaders', 'form2dict', 'parse_form', 'JSONResponse', 'flat_xt',
               'Beforeware', 'EventStream', 'signal_shutdown', 'uri', 'decode_uri', 'flat_tuple', 'noop_body', 'respond',
               'is_full_page', 'Redirect', 'get_key', 'qp', 'def_hdrs', 'FastHTML', 'nested_name', 'serve', 'Client',
               'RouteFuncs', 'APIRouter', 'cookie', 'reg_re_param', 'MiddlewareBase', 'FtResponse', 'unqid']

    # %% ../nbs/api/00_core.ipynb
    def get_function_params(f): 
        return signature_ex(f, True).parameters

    empty = Parameter.empty

    # %% ../nbs/api/00_core.ipynb
    def parsed_date(s:str):
        "Convert `s` to a datetime"
        return dtparse.parse(s)

    # %% ../nbs/api/00_core.ipynb
    def snake2hyphens(s:str):
        "Convert `s` from snake case to hyphenated and capitalised"
        s = snake2camel(s)
        return camel2words(s, '-')

    # %% ../nbs/api/00_core.ipynb
    htmx_hdrs = dict(
        boosted="HX-Boosted",
        current_url="HX-Current-URL",
        history_restore_request="HX-History-Restore-Request",
        prompt="HX-Prompt",
        request="HX-Request",
        target="HX-Target",
        trigger_name="HX-Trigger-Name",
        trigger="HX-Trigger")

    @dataclass
    class HtmxHeaders:
        boosted:str|None=None; current_url:str|None=None; history_restore_request:str|None=None; prompt:str|None=None
        request:str|None=None; target:str|None=None; trigger_name:str|None=None; trigger:str|None=None
        def __bool__(self): return any(hasattr(self,o) for o in htmx_hdrs)

    def get_htmx_headers(h):
        res = {k:h.get(v.lower(), None) for k,v in htmx_hdrs.items()}
        return HtmxHeaders(**res)

    # %% ../nbs/api/00_core.ipynb
    def make_list(t, v): 
        return [t(o) for o in listify(v)]

    # %% ../nbs/api/00_core.ipynb
    fh_cfg = AttrDict(indent=True)

    # %% ../nbs/api/00_core.ipynb
    def fix_annotation(t, o):
        "Create appropriate callable type for casting a `str` to type `t` (or first type in `t` if union)"
        origin = get_origin(t)
        if origin is Union or origin is UnionType or origin in (list,List):
            t = first(o for o in get_args(t) if o!=type(None))
        d = {bool: str2bool, int: str2int, date: str2date, UploadFile: noop}
        res = d.get(t, t)
        if origin in (list,List): return make_list(res, o)
        if not isinstance(o, (str,list,tuple)): return o
        return res(o[-1]) if isinstance(o,(list,tuple)) else res(o)

    # %% ../nbs/api/00_core.ipynb
    def form_arg(k, v, d):
        "Get type by accessing key `k` from `d`, and use to cast `v`"
        if v is None: return
        if not isinstance(v, (str,list,tuple)): return v
        # This is the type we want to cast `v` to
        anno = d.get(k, None)
        if not anno: return v
        return fix_annotation(anno, v)

    # %% ../nbs/api/00_core.ipynb
    @dataclass
    class HttpHeader: k:str;v:str

    # %% ../nbs/api/00_core.ipynb
    def to_htmx_header(s): 
        return 'HX-' + s.replace('_', '-').title()

    htmx_resps = dict(location=None, push_url=None, redirect=None, refresh=None, replace_url=None,
                     reswap=None, retarget=None, reselect=None, trigger=None, trigger_after_settle=None, trigger_after_swap=None)

    # %% ../nbs/api/00_core.ipynb
    @use_kwargs_dict(**htmx_resps)
    def HtmxResponseHeaders(**kwargs):
        "HTMX response headers"
        res = tuple(HttpHeader(to_htmx_header(k), v) for k,v in kwargs.items())
        return res[0] if len(res)==1 else res

    # %% ../nbs/api/00_core.ipynb
    def get_annotations_safe(anno):
        "Same as `get_annotations`, but also works on namedtuples"
        if is_namedtuple(anno): return {o:str for o in anno._fields}
        return get_annotations(anno)

    # %% ../nbs/api/00_core.ipynb
    def is_body_type(anno): 
        return issubclass(anno, (dict,ns)) or hasattr(anno,'__from_request__') or get_annotations_safe(anno)

    # %% ../nbs/api/00_core.ipynb
    def get_form_item(form, k):
        "Return single item `k` from `form` if len 1, otherwise return list"
        if isinstance(form, dict): return form.get(k)
        o = form.getlist(k)
        return o[0] if len(o) == 1 else o if o else None

    # %% ../nbs/api/00_core.ipynb
    def form2dict(form: FormData) -> dict:
        "Convert starlette form data to a dict"
        if isinstance(form, dict): return form
        return {k: get_form_item(form, k) for k in form}

    # %% ../nbs/api/00_core.ipynb
    async def parse_form(req: Request) -> FormData:
        "Starlette errors on empty multipart forms, so this checks for that situation"
        ctype = req.headers.get("Content-Type", "")
        if ctype=='application/json': return await req.json()
        if not ctype.startswith("multipart/form-data"): return await req.form()
        try: boundary = ctype.split("boundary=")[1].strip()
        except IndexError: raise HTTPException(400, "Invalid form-data: no boundary")
        min_len = len(boundary) + 6
        clen = int(req.headers.get("Content-Length", "0"))
        if clen <= min_len: return FormData()
        return await req.form()

    # %% ../nbs/api/00_core.ipynb
    async def from_body(req, p):
        "Parse request form/query data and create an instance of the annotated parameter type"
        anno = p.annotation
        data = form2dict(await parse_form(req))
        if req.query_params: data = {**data, **dict(req.query_params)}
        ctor = getattr(anno, '__from_request__', None)
        if ctor: return await ctor(data, req) if asyncio.iscoroutinefunction(ctor) else ctor(data, req)
        d = get_annotations_safe(anno)
        cargs = {k: form_arg(k, v, d) for k, v in data.items() if not d or k in d}
        return anno(**cargs)

    # %% ../nbs/api/00_core.ipynb
    class JSONResponse(JSONResponseOrig):
        "Same as starlette's version, but auto-stringifies non serializable types"
        def render(self, content: Any) -> bytes:
            res = json.dumps(content, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":"), default=str)
            return res.encode("utf-8")

    # %% ../nbs/api/00_core.ipynb
    # async def find_param(req, arg:str, p:Parameter):
    #     "In `req` find param named `arg` of type in `p` (`arg` is ignored for body types)"
    #     anno = p.annotation
    #     # If there's an annotation of special types, return object of that type
    #     # GenericAlias is a type of typing for iterators like list[int] that is not a class
    #     if isinstance(anno, type) and not isinstance(anno, GenericAlias):
    #         if issubclass(anno, Request): return req
    #         if issubclass(anno, HtmxHeaders): return get_htmx_headers(req.headers)
    #         if issubclass(anno, Starlette): return req.scope['app']
    #         if is_body_type(anno) and 'session'.startswith(arg.lower()): return req.scope.get('session', {})
    #         if is_body_type(anno): return await from_body(req, p)
    #     # If there's no annotation, check for special names
    #     if anno is empty:
    #         if 'request'.startswith(arg.lower()): return req
    #         if 'session'.startswith(arg.lower()): return req.scope.get('session', {})
    #         if arg.lower()=='scope': return dict2obj(req.scope)
    #         if arg.lower()=='auth': return req.scope.get('auth', None)
    #         if arg.lower()=='htmx': return get_htmx_headers(req.headers)
    #         if arg.lower()=='app': return req.scope['app']
    #         if arg.lower()=='body': return (await req.body()).decode()
    #         if arg.lower() in ('hdrs','ftrs','bodykw','htmlkw'): return getattr(req, arg.lower())
    #         if arg!='resp': warn(f"`{arg} has no type annotation and is not a recognised special name, so is ignored.")
    #         return None
    #     # Look through path, cookies, headers, query, and body in that order
    #     res = req.path_params.get(arg, None)
    #     if res in (empty,None): res = req.cookies.get(arg, None)
    #     if res in (empty,None): res = req.headers.get(snake2hyphens(arg), None)
    #     if res in (empty,None): res = req.query_params.getlist(arg)
    #     if res==[]: res = None
    #     if res in (empty,None): res = get_form_item(await parse_form(req), arg)
    #     # Raise 400 error if the param does not include a default
    #     if (res in (empty,None)) and p.default is empty: raise HTTPException(400, f"Missing required field: {arg}")
    #     # If we have a default, return that if we have no value
    #     if res in (empty,None): res = p.default
    #     # We can cast str and list[str] to types; otherwise just return what we have
    #     if anno is empty: return res
    #     try: return fix_annotation(anno, res)
    #     except ValueError: raise HTTPException(404, req.url.path) from None
    async def find_param(req, arg:str, p:Parameter):
        "In `req` find param named `arg` of type in `p` (`arg` is ignored for body types)"
        anno = p.annotation
        # If there's an annotation of special types, return object of that type
        # GenericAlias is a type of typing for iterators like list[int] that is not a class
        if isinstance(anno, type) and not isinstance(anno, GenericAlias):
            if issubclass(anno, Request): return req
            if issubclass(anno, HtmxHeaders): return _get_htmx(req.headers)
            if issubclass(anno, Starlette): return req.scope['app']
            if is_body_type(anno) and 'session'.startswith(arg.lower()): return req.scope.get('session', {})
            if is_body_type(anno): return await _from_body(req, p)
        # If there's no annotation, check for special names
        if anno is empty:
            if 'request'.startswith(arg.lower()): return req
            if 'session'.startswith(arg.lower()): return req.scope.get('session', {})
            if arg.lower()=='scope': return dict2obj(req.scope)
            if arg.lower()=='auth': return req.scope.get('auth', None)
            if arg.lower()=='htmx': return _get_htmx(req.headers)
            # added ds signal detection
            if arg.lower()=='signals':
                print(f"Method: {req.method}")
                print(f"Query params: {req.query_params}")
                if req.method == 'GET':
                    return json.loads(req.query_params.get('datastar', '{}'))
                else:
                    return json.loads((await req.body()).decode())
            if arg.lower()=='app': return req.scope['app']
            if arg.lower()=='body': return (await req.body()).decode()
            if arg.lower() in ('hdrs','ftrs','bodykw','htmlkw'): return getattr(req, arg.lower())
            if arg!='resp': warn(f"`{arg} has no type annotation and is not a recognised special name, so is ignored.")
            return None
        # Look through path, cookies, headers, query, and body in that order
        res = req.path_params.get(arg, None)
        if res in (empty,None): res = req.cookies.get(arg, None)
        if res in (empty,None): res = req.headers.get(snake2hyphens(arg), None)
        if res in (empty,None): res = req.query_params.getlist(arg)
        if res==[]: res = None
        if res in (empty,None): res = _formitem(await parse_form(req), arg)
        # Raise 400 error if the param does not include a default
        if (res in (empty,None)) and p.default is empty: raise HTTPException(400, f"Missing required field: {arg}")
        # If we have a default, return that if we have no value
        if res in (empty,None): res = p.default
        # We can cast str and list[str] to types; otherwise just return what we have
        if anno is empty: return res
        try: return _fix_anno(anno, res)
        except ValueError: raise HTTPException(404, req.url.path) from None


    async def wrap_request(req, params):
        return {arg:await find_param(req, arg, p) for arg,p in params.items()}

    # %% ../nbs/api/00_core.ipynb
    def flat_xt(lst):
        "Flatten lists"
        result = []
        if isinstance(lst,(FT,str)): lst=[lst]
        for item in lst:
            if isinstance(item, (list,tuple)): result.extend(item)
            else: result.append(item)
        return tuple(result)

    # %% ../nbs/api/00_core.ipynb
    class Beforeware:
        def __init__(self, f, skip=None): self.f,self.skip = f,skip or []

    # %% ../nbs/api/00_core.ipynb
    async def handle_async(f, *args, **kwargs):
        return (await f(*args, **kwargs)) if is_async_callable(f) else await run_in_threadpool(f, *args, **kwargs)

    # %% ../nbs/api/00_core.ipynb
    def find_websocket_param(ws, data, hdrs, arg:str, p:Parameter):
        "In `data` find param named `arg` of type in `p` (`arg` is ignored for body types)"
        anno = p.annotation
        if isinstance(anno, type):
            if issubclass(anno, HtmxHeaders): return get_htmx_headers(hdrs)
            if issubclass(anno, Starlette): return ws.scope['app']
            if issubclass(anno, WebSocket): return ws
            if issubclass(anno, dict): return data
        if anno is empty:
            if arg.lower()=='ws': return ws
            if arg.lower()=='scope': return dict2obj(ws.scope)
            if arg.lower()=='data': return data
            if arg.lower()=='htmx': return get_htmx_headers(hdrs)
            if arg.lower()=='app': return ws.scope['app']
            if arg.lower()=='send': return partial(send_websocket, ws)
            if 'session'.startswith(arg.lower()): return ws.scope.get('session', {})
            return None
        res = data.get(arg, None)
        if res is empty or res is None: res = hdrs.get(arg, None)
        if res is empty or res is None: res = p.default
        # We can cast str and list[str] to types; otherwise just return what we have
        if not isinstance(res, (list,str)) or anno is empty: return res
        return [fix_annotation(anno, o) for o in res] if isinstance(res,list) else fix_annotation(anno, res)

    def wrap_websocket(ws, data, params):
        hdrs = {k.lower().replace('-','_'):v for k,v in data.pop('HEADERS', {}).items()}
        return {arg:find_websocket_param(ws, data, hdrs, arg, p) for arg,p in params.items()}

    # %% ../nbs/api/00_core.ipynb
    async def send_websocket(ws, resp):
        if not resp: return
        res = to_xml(resp, indent=fh_cfg.indent)
        await ws.send_text(res)

    def websocket_endpoint(recv, conn=None, disconn=None):
        cls = type('WS_Endp', (WebSocketEndpoint,), {"encoding":"text"})

        async def generic_handler(handler, ws, data=None):
            wd = wrap_websocket(ws, loads(data) if data else {}, get_function_params(handler))
            resp = await handle_async(handler, **wd)
            if resp: await send_websocket(ws, resp)

        async def connect_handler(self, ws):
            await ws.accept()
            await generic_handler(conn, ws)

        async def disconnect_handler(self, ws, close_code): 
            await generic_handler(disconn, ws)

        async def receive_handler(self, ws, data): 
            await generic_handler(recv, ws, data)

        if conn: cls.on_connect = connect_handler
        if disconn: cls.on_disconnect = disconnect_handler
        cls.on_receive = receive_handler
        return cls

    # %% ../nbs/api/00_core.ipynb
    def EventStream(s):
        "Create a text/event-stream response from `s`"
        return StreamingResponse(s, media_type="text/event-stream")

    # %% ../nbs/api/00_core.ipynb
    def signal_shutdown():
        from uvicorn.main import Server
        event = asyncio.Event()
        @patch
        def handle_exit(self:Server, *args, **kwargs):
            event.set()
            self.force_exit = True
            self.orig_handle_exit(*args, **kwargs)
        return event

    # %% ../nbs/api/00_core.ipynb
    def uri(arg, **kwargs):
        "Create a URI by URL-encoding `arg` and appending query parameters from `kwargs`"
        return f"{quote(arg)}/{urlencode(kwargs, doseq=True)}"

    # %% ../nbs/api/00_core.ipynb
    def decode_uri(s):
        "Decode a URI created by `uri()` back into argument and keyword dict"
        arg,_,kw = s.partition('/')
        return unquote(arg), {k:v[0] for k,v in parse_qs(kw).items()}

    # %% ../nbs/api/00_core.ipynb
    StringConvertor.regex = "[^/]*"  # `+` replaced with `*`

    @patch
    def to_string(self:StringConvertor, value: str) -> str:
        value = str(value)
        assert "/" not in value, "May not contain path separators"
        # assert value, "Must not be empty"  # line removed due to errors
        return value

    # %% ../nbs/api/00_core.ipynb
    @patch
    def url_path_for(self:HTTPConnection, name: str, **path_params):
        lp = self.scope['app'].url_path_for(name, **path_params)
        return URLPath(f"{self.scope['root_path']}{lp}", lp.protocol, lp.host)

    # %% ../nbs/api/00_core.ipynb
    verbs = dict(get='hx-get', post='hx-post', put='hx-post', delete='hx-delete', patch='hx-patch', link='href')

    def url_for(req, t):
        "Generate URL for route `t` using request `req`"
        if callable(t): t = t.__routename__
        kw = {}
        if t.find('/')>-1 and (t.find('?')<0 or t.find('/')<t.find('?')): t,kw = decode_uri(t)
        t,m,q = t.partition('?')
        return f"{req.url_path_for(t, **kw)}{m}{q}"

    def find_targets(req, resp):
        "Find and convert route targets in response attributes to URLs"
        if isinstance(resp, tuple):
            for o in resp: find_targets(req, o)
        if isinstance(resp, FT):
            for o in resp.children: find_targets(req, o)
            for k,v in verbs.items():
                t = resp.attrs.pop(k, None)
                if t: resp.attrs[v] = url_for(req, t)

    def apply_ft(o):
        "Apply FastTag transformation recursively to object `o`"
        if isinstance(o, tuple): o = tuple(apply_ft(c) for c in o)
        if hasattr(o, '__ft__'): o = o.__ft__()
        if isinstance(o, FT): o.children = tuple(apply_ft(c) for c in o.children)
        return o

    def to_xml_with_targets(req, resp, indent):
        "Convert response to XML string with target URL resolution"
        resp = apply_ft(resp)
        find_targets(req, resp)
        return to_xml(resp, indent)

    # %% ../nbs/api/00_core.ipynb
    iter_types = (tuple,list,map,filter,range,types.GeneratorType)

    # %% ../nbs/api/00_core.ipynb
    def flat_tuple(o):
        "Flatten nested iterables into a single tuple"
        result = []
        if not isinstance(o,iter_types): o=[o]
        o = list(o)
        for item in o:
            if isinstance(item, iter_types): result.extend(list(item))
            else: result.append(item)
        return tuple(result)

    # %% ../nbs/api/00_core.ipynb
    def noop_body(c, req):
        "Default Body wrap function which just returns the content"
        return c

    # %% ../nbs/api/00_core.ipynb
    def respond(req, heads, bdy):
        "Default FT response creation function"
        body_wrap = getattr(req, 'body_wrap', noop_body)
        params = inspect.signature(body_wrap).parameters
        bw_args = (bdy, req) if len(params)>1 else (bdy,)
        body = Body(body_wrap(*bw_args), *flat_xt(req.ftrs), **req.bodykw)
        return Html(Head(*heads, *flat_xt(req.hdrs)), body, **req.htmlkw)

    # %% ../nbs/api/00_core.ipynb
    def is_full_page(req, resp):
        "Check if response should be rendered as full page or fragment"
        if resp and any(getattr(o, 'tag', '')=='html' for o in resp): return True
        return 'hx-request' in req.headers and 'hx-history-restore-request' not in req.headers

    # %% ../nbs/api/00_core.ipynb
    def partition_response(req, resp):
        "Partition response into HTTP headers, background tasks, and content"
        resp = flat_tuple(resp)
        resp = resp + tuple(getattr(req, 'injects', ()))
        http_hdrs,resp = partition(resp, risinstance(HttpHeader))
        tasks,resp = partition(resp, risinstance(BackgroundTask))
        kw = {"headers": {"vary": "HX-Request, HX-History-Restore-Request"}}
        if http_hdrs: kw['headers'] |= {o.k:str(o.v) for o in http_hdrs}
        if tasks:
            ts = BackgroundTasks()
            for t in tasks: ts.tasks.append(t)
            kw['background'] = ts
        resp = tuple(resp)
        if len(resp)==1: resp = resp[0]
        return resp,kw

    # %% ../nbs/api/00_core.ipynb
    def extract_content(req, resp):
        "Extract content and headers, render as full page or fragment"
        hdr_tags = 'title','meta','link','style','base'
        resp = tuplify(resp)
        heads,bdy = partition(resp, lambda o: getattr(o, 'tag', '') in hdr_tags)
        if not is_full_page(req, resp):
            title = [] if any(getattr(o, 'tag', '')=='title' for o in heads) else [Title(req.app.title)]
            canonical = [Link(rel="canonical", href=getattr(req, 'canonical', req.url))] if req.app.canonical else []
            resp = respond(req, [*heads, *title, *canonical], bdy)
        return to_xml_with_targets(req, resp, indent=fh_cfg.indent)

    # %% ../nbs/api/00_core.ipynb
    def is_ft_response(resp):
        "Check if response is a FastTag-compatible type"
        return isinstance(resp, iter_types+(HttpHeader,FT)) or hasattr(resp, '__ft__')

    # %% ../nbs/api/00_core.ipynb
    def create_response(req, resp, cls=empty, status_code=200):
        "Create appropriate HTTP response from request and response data"
        if not resp: resp=''
        if hasattr(resp, '__response__'): resp = resp.__response__(req)
        if cls in (Any,FT): cls=empty
        if isinstance(resp, FileResponse) and not os.path.exists(resp.path): raise HTTPException(404, resp.path)
        resp,kw = partition_response(req, resp)
        if cls is not empty: return cls(resp, status_code=status_code, **kw)
        if isinstance(resp, Response): return resp
        # added datastar handeling for SSE here. 
        if is_ft_response(resp):
            if req.headers.get('datastar-request') == 'true':
                # Format as SSE using your build_sse_event function
                sse_content = build_sse_event(resp)
                return StreamingResponse(iter([sse_content]), media_type="text/event-stream", **kw)
            else:
                cts = extract_content(req, resp)
                return HTMLResponse(cts, status_code=status_code, **kw)


        if isinstance(resp, str): cls = HTMLResponse
        elif isinstance(resp, Mapping): cls = JSONResponse
        else:
            resp = str(resp)
            cls = HTMLResponse
        return cls(resp, status_code=status_code, **kw)

    # %% ../nbs/api/00_core.ipynb
    class Redirect:
        "Use HTMX or Starlette RedirectResponse as required to redirect to `loc`"
        def __init__(self, loc): self.loc = loc
        def __response__(self, req):
            if 'hx-request' in req.headers: return HtmxResponseHeaders(redirect=self.loc)
            return RedirectResponse(self.loc, status_code=303)

    # %% ../nbs/api/00_core.ipynb
    async def wrap_function_call(f, req, params):
        "Wrap function call with request parameter injection"
        wreq = await wrap_request(req, params)
        return await handle_async(f, **wreq)

    # %% ../nbs/api/00_core.ipynb
    htmx_exts = {
        "morph": "https://cdn.jsdelivr.net/npm/idiomorph@0.7.3/dist/idiomorph-ext.min.js",
        "head-support": "https://cdn.jsdelivr.net/npm/htmx-ext-head-support@2.0.4/head-support.js",
        "preload": "https://cdn.jsdelivr.net/npm/htmx-ext-preload@2.1.1/preload.js",
        "class-tools": "https://cdn.jsdelivr.net/npm/htmx-ext-class-tools@2.0.1/class-tools.js",
        "loading-states": "https://cdn.jsdelivr.net/npm/htmx-ext-loading-states@2.0.1/loading-states.js",
        "multi-swap": "https://cdn.jsdelivr.net/npm/htmx-ext-multi-swap@2.0.0/multi-swap.js",
        "path-deps": "https://cdn.jsdelivr.net/npm/htmx-ext-path-deps@2.0.0/path-deps.js",
        "remove-me": "https://cdn.jsdelivr.net/npm/htmx-ext-remove-me@2.0.0/remove-me.js",
        "ws": "https://cdn.jsdelivr.net/npm/htmx-ext-ws@2.0.3/ws.js",
        "chunked-transfer": "https://cdn.jsdelivr.net/npm/htmx-ext-transfer-encoding-chunked@0.4.0/transfer-encoding-chunked.js"
    }

    # %% ../nbs/api/00_core.ipynb
    datastarsrc = Script(type="module", src="https://cdn.jsdelivr.net/gh/starfederation/datastar@1.0.0-RC.5/bundles/datastar.js")
    htmxsrc     = Script(src="https://cdn.jsdelivr.net/npm/htmx.org@2.0.6/dist/htmx.min.js")
    fhjsscr     = Script(src="https://cdn.jsdelivr.net/gh/answerdotai/fasthtml-js@1.0.12/fasthtml.js")
    surrsrc     = Script(src="https://cdn.jsdelivr.net/gh/answerdotai/surreal@main/surreal.js")
    scopesrc    = Script(src="https://cdn.jsdelivr.net/gh/gnat/css-scope-inline@main/script.js")
    viewport    = Meta(name="viewport", content="width=device-width, initial-scale=1, viewport-fit=cover")
    charset     = Meta(charset="utf-8")

    # %% ../nbs/api/00_core.ipynb
    def get_key(key=None, fname='.sesskey'):
        "Get session key from `key` param or read/create from file `fname`"
        if key: return key
        fname = Path(fname)
        if fname.exists(): return fname.read_text()
        key = str(uuid.uuid4())
        fname.write_text(key)
        return key

    # %% ../nbs/api/00_core.ipynb
    def make_list_safe(o):
        "Wrap non-list item in a list, returning empty list if None"
        return [] if not o else list(o) if isinstance(o, (tuple,list)) else [o]

    # %% ../nbs/api/00_core.ipynb
    def wrap_exception_handler(f, status_code, hdrs, ftrs, htmlkw, bodykw, body_wrap):
        "Wrap exception handler with FastHTML request processing"
        async def handler_func(req, exc):
            req.hdrs,req.ftrs,req.htmlkw,req.bodykw = map(deepcopy, (hdrs, ftrs, htmlkw, bodykw))
            req.body_wrap = body_wrap
            res = await handle_async(f, req, exc)
            return create_response(req, res, status_code=status_code)
        return handler_func

    # %% ../nbs/api/00_core.ipynb
    def qp(p:str, **kw) -> str:
        "Add parameters kw to path p"
        def substitute_param(m):
            pre,post = m.groups()
            if pre not in kw: return f'{{{pre}{post or ""}}}'
            pre = kw.pop(pre)
            return '' if pre in (False,None) else str(pre)
        p = re.sub(r'\{([^:}]+)(:.+?)?}', substitute_param, p)
        # encode query params
        return p + ('?' + urlencode({k:'' if v in (False,None) else v for k,v in kw.items()},doseq=True) if kw else '')

    # %% ../nbs/api/00_core.ipynb
    def def_hdrs(htmx=False, surreal=False, datastar=True):
        "Default headers for a FastHTML app (mod. for datastar)"
        hdrs = []
        if surreal: hdrs = [surrsrc,scopesrc] + hdrs
        if htmx: hdrs = [htmxsrc,fhjsscr] + hdrs
        if datastar: hdrs = [datastarsrc] + hdrs
        return [charset, viewport] + hdrs

    # %% ../nbs/api/00_core.ipynb
    cors_allow = Middleware(CORSMiddleware, allow_credentials=True,
                            allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    iframe_scr = Script(NotStr("""
        function sendmsg() {
            window.parent.postMessage({height: document.documentElement.offsetHeight}, '*');
        }
        window.onload = function() {
            sendmsg();
            document.body.addEventListener('htmx:afterSettle',    sendmsg);
            document.body.addEventListener('htmx:wsAfterMessage', sendmsg);
        };"""))

    # %% ../nbs/api/00_core.ipynb
    class FastHTML(Starlette):
        def __init__(self, debug=False, routes=None, middleware=None, title: str = "FastTag + Datastar", exception_handlers=None,
                     on_startup=None, on_shutdown=None, lifespan=None, hdrs=None, ftrs=None, exts=None,
                     before=None, after=None, surreal=False, htmx=False, datastar=True,  
                     default_hdrs=True, sess_cls=SessionMiddleware,
                     secret_key=None, session_cookie='session_', max_age=365*24*3600, sess_path='/',
                     same_site='lax', sess_https_only=False, sess_domain=None, key_fname='.sesskey',
                     body_wrap=noop_body, htmlkw=None, nb_hdrs=False, canonical=True, **bodykw):
            middleware,before,after = map(make_list_safe, (middleware,before,after))
            self.title,self.canonical = title,canonical
            hdrs,ftrs,exts = map(listify, (hdrs,ftrs,exts))
            exts = {k:htmx_exts[k] for k in exts}
            htmlkw = htmlkw or {}
            if default_hdrs: hdrs = def_hdrs(htmx, surreal=surreal, datastar=datastar ) + hdrs
            hdrs += [Script(src=ext) for ext in exts.values()]
            if IN_NOTEBOOK:
                hdrs.append(iframe_scr)
                from IPython.display import display,HTML
                if nb_hdrs: display(HTML(to_xml(tuple(hdrs))))
                middleware.append(cors_allow)
            on_startup,on_shutdown = listify(on_startup) or None,listify(on_shutdown) or None
            self.lifespan,self.hdrs,self.ftrs = lifespan,hdrs,ftrs
            self.body_wrap,self.before,self.after,self.htmlkw,self.bodykw = body_wrap,before,after,htmlkw,bodykw
            secret_key = get_key(secret_key, key_fname)
            if sess_cls:
                sess = Middleware(sess_cls, secret_key=secret_key,session_cookie=session_cookie,
                                  max_age=max_age, path=sess_path, same_site=same_site,
                                  https_only=sess_https_only, domain=sess_domain)
                middleware.append(sess)
            exception_handlers = ifnone(exception_handlers, {})
            if 404 not in exception_handlers:
                def not_found_handler(req, exc): return  Response('404 Not Found', status_code=404)
                exception_handlers[404] = not_found_handler
            excs = {k:wrap_exception_handler(v, k, hdrs, ftrs, htmlkw, bodykw, body_wrap=body_wrap) for k,v in exception_handlers.items()}
            super().__init__(debug, routes, middleware=middleware, exception_handlers=excs, on_startup=on_startup, on_shutdown=on_shutdown, lifespan=lifespan)

    # %% ../nbs/api/00_core.ipynb
    @patch
    def add_route(self:FastHTML, route):
        "Add or replace a route in the FastHTML app"
        route.methods = [m.upper() for m in listify(route.methods)]
        self.router.routes = [r for r in self.router.routes if not
                       (r.path==route.path and r.name == route.name and
                        ((route.methods is None) or (set(r.methods) == set(route.methods))))]
        self.router.routes.append(route)

    # %% ../nbs/api/00_core.ipynb
    all_meths = 'get post put delete patch head trace options'.split()

    # %% ../nbs/api/00_core.ipynb
    @patch
    def create_endpoint(self:FastHTML, f, body_wrap):
        "Create endpoint wrapper with before/after middleware processing"
        sig = signature_ex(f, True)
        async def endpoint_func(req):
            resp = None
            req.injects = []
            req.hdrs,req.ftrs,req.htmlkw,req.bodykw = map(deepcopy, (self.hdrs,self.ftrs,self.htmlkw,self.bodykw))
            req.hdrs,req.ftrs = listify(req.hdrs),listify(req.ftrs)
            for b in self.before:
                if not resp:
                    if isinstance(b, Beforeware): bf,skip = b.f,b.skip
                    else: bf,skip = b,[]
                    if not any(re.fullmatch(r, req.url.path) for r in skip):
                        resp = await wrap_function_call(bf, req, get_function_params(bf))
            req.body_wrap = body_wrap
            if not resp: resp = await wrap_function_call(f, req, sig.parameters)
            for a in self.after:
                wreq = await wrap_request(req, get_function_params(a))
                wreq['resp'] = resp
                nr = a(**wreq)
                if nr: resp = nr
            return create_response(req, resp, sig.return_annotation)
        return endpoint_func

    # %% ../nbs/api/00_core.ipynb
    @patch
    def add_websocket_route(self:FastHTML, func, path, conn, disconn, name, middleware):
        "Add websocket route to FastHTML app"
        endp = websocket_endpoint(func, conn, disconn)
        route = WebSocketRoute(path, endpoint=endp, name=name, middleware=middleware)
        route.methods = ['ws']
        self.add_route(route)
        return func

    # %% ../nbs/api/00_core.ipynb
    @patch
    def ws(self:FastHTML, path:str, conn=None, disconn=None, name=None, middleware=None):
        "Add a websocket route at `path`"
        def f(func=noop): return self.add_websocket_route(func, path, conn, disconn, name=name, middleware=middleware)
        return f

    # %% ../nbs/api/00_core.ipynb
    def make_location_function(f,p):
        "Create a location function wrapper with route path and to() method"
        class LocationFunction:
            def __init__(self): update_wrapper(self, f)
            def __call__(self, *args, **kw): return f(*args, **kw)
            def to(self, **kw): return qp(p, **kw)
            def __str__(self): return p
        return LocationFunction()

    # %% ../nbs/api/00_core.ipynb
    def nested_name(f):
        "Get name of function `f` using '_' to join nested function names"
        return f.__qualname__.replace('.<locals>.', '_')

    # %% ../nbs/api/00_core.ipynb
    @patch
    def add_http_route(self:FastHTML, func, path, methods, name, include_in_schema, body_wrap):
        "Add HTTP route to FastHTML app with automatic method detection"
        n,fn,p = name,nested_name(func),None if callable(path) else path
        if methods: m = [methods] if isinstance(methods,str) else methods
        elif fn in all_meths and p is not None: m = [fn]
        else: m = ['get','post']
        if not n: n = fn
        if not p: p = '/'+('' if fn=='index' else fn)
        route = Route(p, endpoint=self.create_endpoint(func, body_wrap or self.body_wrap), methods=m, name=n, include_in_schema=include_in_schema)
        self.add_route(route)
        lf = make_location_function(func, p)
        lf.__routename__ = n
        return lf

    # %% ../nbs/api/00_core.ipynb
    @patch
    def route(self:FastHTML, path:str=None, methods=None, name=None, include_in_schema=True, body_wrap=None):
        "Add a route at `path`"
        def f(func): return self.add_http_route(func, path, methods, name=name, include_in_schema=include_in_schema, body_wrap=body_wrap)
        return f(path) if callable(path) else f

    for o in all_meths: setattr(FastHTML, o, partialmethod(FastHTML.route, methods=o))

    # %% ../nbs/api/00_core.ipynb
    @patch
    def set_lifespan(self:FastHTML, value):
        "Set the lifespan context manager for the FastHTML app"
        if inspect.isasyncgenfunction(value): value = contextlib.asynccontextmanager(value)
        self.router.lifespan_context = value

    # %% ../nbs/api/00_core.ipynb
    def serve(
            appname=None, # Name of the module
            app='app', # App instance to be served
            host='0.0.0.0', # If host is 0.0.0.0 will convert to localhost
            port=None, # If port is None it will default to 5001 or the PORT environment variable
            reload=True, # Default is to reload the app upon code changes
            reload_includes:list[str]|str|None=None, # Additional files to watch for changes
            reload_excludes:list[str]|str|None=None # Files to ignore for changes
            ):
        "Run the app in an async server, with live reload set as the default."
        bk = inspect.currentframe().f_back
        glb = bk.f_globals
        code = bk.f_code
        if not appname:
            if glb.get('__name__')=='__main__': appname = Path(glb.get('__file__', '')).stem
            elif code.co_name=='main' and bk.f_back.f_globals.get('__name__')=='__main__': appname = inspect.getmodule(bk).__name__
        import uvicorn
        if appname:
            if not port: port=int(os.getenv("PORT", default=5001))
            print(f'Link: http://{"localhost" if host=="0.0.0.0" else host}:{port}')
            uvicorn.run(f'{appname}:{app}', host=host, port=port, reload=reload, reload_includes=reload_includes, reload_excludes=reload_excludes)

    # %% ../nbs/api/00_core.ipynb
    class Client:
        "A simple httpx ASGI client that doesn't require `async`"
        def __init__(self, app, url="http://testserver"):
            self.cli = AsyncClient(transport=ASGITransport(app), base_url=url)

        def sync_request(self, method, url, **kwargs):
            async def request(): return await self.cli.request(method, url, **kwargs)
            with from_thread.start_blocking_portal() as portal: return portal.call(request)

    for o in ('get', 'post', 'delete', 'put', 'patch', 'options'): setattr(Client, o, partialmethod(Client.sync_request, o))

    # %% ../nbs/api/00_core.ipynb
    class RouteFuncs:
        def __init__(self): super().__setattr__('route_functions', {})
        def __setattr__(self, name, value): self.route_functions[name] = value
        def __getattr__(self, name):
            if name in all_meths: raise AttributeError("Route functions with HTTP Names are not accessible here")
            try: return self.route_functions[name]
            except KeyError: raise AttributeError(f"No route named {name} found in route functions")
        def __dir__(self): return list(self.route_functions.keys())

    # %% ../nbs/api/00_core.ipynb
    class APIRouter:
        "Add routes to an app"
        def __init__(self, prefix:str|None=None, body_wrap=noop_body):
            self.routes,self.wss = [],[]
            self.rt_funcs = RouteFuncs()  # Store wrapped route function for discoverability
            self.prefix = prefix if prefix else ""
            self.body_wrap = body_wrap

        def wrap_route_function(self, func, path=None):
            name = func.__name__
            wrapped = make_location_function(func, path)
            wrapped.__routename__ = name
            # If you are using the def get or def post method names, this approach is not supported
            if name not in all_meths: setattr(self.rt_funcs, name, wrapped)
            return wrapped

        def __call__(self, path:str=None, methods=None, name=None, include_in_schema=True, body_wrap=None):
            "Add a route at `path`"
            def f(func):
                p = self.prefix + ("/" + ('' if path.__name__=='index' else func.__name__) if callable(path) else path)
                wrapped = self.wrap_route_function(func, p)
                self.routes.append((func, p, methods, name, include_in_schema, body_wrap or self.body_wrap))
                return wrapped
            return f(path) if callable(path) else f

        def __getattr__(self, name):
            try: return getattr(self.rt_funcs, name)
            except AttributeError: return super().__getattr__(self, name)

        def to_app(self, app):
            "Add routes to `app`"
            for args in self.routes: app.add_http_route(*args)
            for args in self.wss: app.add_websocket_route(*args)

        def ws(self, path:str, conn=None, disconn=None, name=None, middleware=None):
            "Add a websocket route at `path`"
            def f(func=noop): return self.wss.append((func, f"{self.prefix}{path}", conn, disconn, name, middleware))
            return f

    # %% ../nbs/api/00_core.ipynb
    for o in all_meths: setattr(APIRouter, o, partialmethod(APIRouter.__call__, methods=o))

    # %% ../nbs/api/00_core.ipynb
    def cookie(key: str, value="", max_age=None, expires=None, path="/", domain=None, secure=False, httponly=False, samesite="lax",):
        "Create a 'set-cookie' `HttpHeader`"
        cookie = cookies.SimpleCookie()
        cookie[key] = value
        if max_age is not None: cookie[key]["max-age"] = max_age
        if expires is not None:
            cookie[key]["expires"] = format_datetime(expires, usegmt=True) if isinstance(expires, datetime) else expires
        if path is not None: cookie[key]["path"] = path
        if domain is not None: cookie[key]["domain"] = domain
        if secure: cookie[key]["secure"] = True
        if httponly: cookie[key]["httponly"] = True
        if samesite is not None:
            assert samesite.lower() in [ "strict", "lax", "none", ], "must be 'strict', 'lax' or 'none'"
            cookie[key]["samesite"] = samesite
        cookie_val = cookie.output(header="").strip()
        return HttpHeader("set-cookie", cookie_val)

    # %% ../nbs/api/00_core.ipynb
    def reg_re_param(m, s):
        cls = get_class(f'{m}Conv', sup=StringConvertor, regex=s)
        register_url_convertor(m, cls())

    # %% ../nbs/api/00_core.ipynb
    # Starlette doesn't have the '?', so it chomps the whole remaining URL
    reg_re_param("path", ".*?")
    static_extensions = "ico gif jpg jpeg webm css js woff png svg mp4 webp ttf otf eot woff2 txt html map pdf zip tgz gz csv mp3 wav ogg flac aac doc docx xls xlsx ppt pptx epub mobi bmp tiff avi mov wmv mkv xml yaml yml rar 7z tar bz2 htm xhtml apk dmg exe msi swf iso".split()
    reg_re_param("static", '|'.join(static_extensions))

    @patch
    def static_route_exts(self:FastHTML, prefix='/', static_path='.', exts='static'):
        "Add a static route at URL path `prefix` with files from `static_path` and `exts` defined by `reg_re_param()`"
        @self.route(f"{prefix}{{fname:path}}.{{ext:{exts}}}")
        async def get(fname:str, ext:str): return FileResponse(f'{static_path}/{fname}.{ext}')

    # %% ../nbs/api/00_core.ipynb
    @patch
    def static_route(self:FastHTML, ext='', prefix='/', static_path='.'):
        "Add a static route at URL path `prefix` with files from `static_path` and single `ext` (including the '.')"
        @self.route(f"{prefix}{{fname:path}}{ext}")
        async def get(fname:str): return FileResponse(f'{static_path}/{fname}{ext}')

    # %% ../nbs/api/00_core.ipynb
    class MiddlewareBase:
        async def __call__(self, scope, receive, send) -> None:
            if scope["type"] not in ["http", "websocket"]:
                await self.app(scope, receive, send)
                return
            return HTTPConnection(scope)

    # %% ../nbs/api/00_core.ipynb
    class FtResponse:
        "Wrap an FT response with any Starlette `Response`"
        def __init__(self, content, status_code:int=200, headers=None, cls=HTMLResponse, media_type:str|None=None, background: BackgroundTask | None = None):
            self.content,self.status_code,self.headers = content,status_code,headers
            self.cls,self.media_type,self.background = cls,media_type,background

        def __response__(self, req):
            resp,kw = partition_response(req, self.content)
            cts = extract_content(req, resp)
            tasks,httphdrs = kw.get('background'),kw.get('headers')
            if not tasks: tasks = self.background
            headers = {**(self.headers or {}), **httphdrs}
            return self.cls(cts, status_code=self.status_code, headers=headers, media_type=self.media_type, background=tasks)

    # %% ../nbs/api/00_core.ipynb
    def unqid(seeded=False):
        id4 = UUID(int=random.getrandbits(128), version=4) if seeded else uuid4()
        res = b64encode(id4.bytes)
        return '_' + res.decode().rstrip('=').translate(str.maketrans('+/', '_-'))

    # %% ../nbs/api/00_core.ipynb
    def add_ids(s):
        if not isinstance(s, FT): return
        if not getattr(s, 'id', None): s.id = unqid()
        for c in s.children: add_ids(c)

    # %% ../nbs/api/00_core.ipynb
    @patch
    def setup_ws(app:FastHTML, f=noop):
        conns = {}
        async def on_connect(scope, send): conns[scope.client] = send
        async def on_disconnect(scope): conns.pop(scope.client)
        app.ws('/ws', conn=on_connect, disconn=on_disconnect)(f)
        async def send(s):
            for o in conns.values(): await o(s)
        app.websocket_send = send
        return send

    # %% ../nbs/api/00_core.ipynb
    devtools_loc = "/.well-known/appspecific/com.chrome.devtools.json"

    @patch
    def devtools_json(self:FastHTML, path=None, uuid=None):
        if not path: path = Path().absolute()
        if not uuid: uuid = get_key()
        @self.route(devtools_loc)
        def devtools():
            return dict(workspace=dict(root=path, uuid=uuid))
    return (FastHTML,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Datastar utils""")
    return


@app.cell
def _(to_xml):
    def build_sse_event(ft_content, mode=None, selector=None, transition=False):
        data_lines = ["event: datastar-patch-elements"]
        if mode: data_lines.append(f"data: mode {mode}")
        if selector: data_lines.append(f"data: selector {selector}")
        if transition: data_lines.append("data: useViewTransition true")

        html = to_xml(ft_content, indent=True)
        lines = html.rstrip('\n').split('\n')

        data_lines.extend([f"data: elements {line}" for line in lines])

        return "\n".join(data_lines) + "\n\n"
    return (build_sse_event,)


@app.cell
def _(Div, build_sse_event):
    def test_basic_element_patch():
        result = build_sse_event(Div("Hello world!", id="foo"))
        expected = 'event: datastar-patch-elements\ndata: elements <div id="foo">Hello world!</div>\n\n'
        assert result == expected

    def test_with_mode():
        result = build_sse_event(Div("content"), mode="inner")
        assert "data: mode inner" in result
        assert "event: datastar-patch-elements" in result

    def test_with_selector():
        result = build_sse_event(Div("content"), selector="#target")
        assert "data: selector #target" in result

    def test_with_view_transition():
        result = build_sse_event(Div("content"), transition=True)
        assert "data: useViewTransition true" in result
    return


@app.cell
def _(FastHTML, TestClient, json):
    def test_signals_get_request():
        app = FastHTML()

        @app.get('/test-signals')
        def test_route(signals): 
            return f"Received signals: {signals}"

        client = TestClient(app)
        signal_data = {"user_id": 123, "current_page": "dashboard"}
        response = client.get(f'/test-signals?datastar={json.dumps(signal_data)}')

        assert "{'user_id': 123, 'current_page': 'dashboard'}" in response.text

    def test_signals_post_request():
        app = FastHTML()

        @app.post('/test-signals')
        def test_route(signals): 
            return f"Received signals: {signals}"

        client = TestClient(app)
        signal_data = {"user_id": 456, "action": "submit_form"}
        response = client.post('/test-signals', json=signal_data)

        assert "{'user_id': 456, 'action': 'submit_form'}" in response.text
    return


@app.cell
def _(json):
    def build_signal_event(signals, only_if_missing=False):
        data_lines = ["event: datastar-patch-signals"]
        if only_if_missing:
            data_lines.append("data: onlyIfMissing true")

        # Convert signals dict to JSON string
        signal_json = json.dumps(signals)
        data_lines.append(f"data: signals {signal_json}")

        return "\n".join(data_lines) + "\n\n"
    return (build_signal_event,)


@app.cell
def _(build_signal_event):
    def test_build_signal_event_basic():
        signals = {"validation_errors": [], "form_valid": True}
        result = build_signal_event(signals)

        assert "event: datastar-patch-signals" in result
        assert 'data: signals {"validation_errors": [], "form_valid": true}' in result
        assert result.endswith("\n\n")
    return


@app.cell
def _(Div, FastHTML, TestClient):
    def test_datastar_sse_response():
        app = FastHTML()

        @app.get('/test')
        def test_route():
            return Div("Hello Datastar!", id="content")

        client = TestClient(app)

        # Test without datastar header (should get HTML)
        response = client.get('/test')
        assert response.headers['content-type'] == 'text/html; charset=utf-8'
        assert '<div id="content">Hello Datastar!</div>' in response.text

        # Test with datastar header (should get SSE)
        response_sse = client.get('/test', headers={'datastar-request': 'true'})
        assert 'text/event-stream' in response_sse.headers['content-type']
        assert 'event: datastar-patch-elements' in response_sse.text
        assert 'data: elements <div id="content">Hello Datastar!</div>' in response_sse.text
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## demo (WIP)""")
    return


@app.cell
def _():
    import apsw
    return (apsw,)


@app.cell
def _(apsw, db_path):
    #| export


    class DBPool:
        """
        Connection pool for SQLite database access using APSW.

        Manages a pool of database connections to improve performance by reusing
        connections rather than creating new ones for each operation.
        """

        def __init__(self, 
                     db_path: str = "data/app.db",  # Path to the SQLite database file
                     max_size: int = 6              # Maximum number of connections to keep in pool
                    ):
            """Initialize the database connection pool."""
            self.db_path, self.max_size, self.pool = db_path, max_size, []

        def get_connection(self):
            """
            Get a database connection from the pool or create a new one.

            Returns a connection with 30-second busy timeout configured.
            """
            if self.pool: 
                return self.pool.pop()

            try:
                conn = apsw.Connection(self.db_path)
                conn.setbusytimeout(30000)  # 30 second timeout
                return conn
            except Exception as e:
                print(f"Failed to create connection: {e}")
                raise

        def return_connection(self, 
                             conn  # Database connection to return to pool
                            ):
            """Return a connection to the pool or close it if pool is full."""
            if len(self.pool) < self.max_size: 
                self.pool.append(conn)
            else: 
                conn.close()

        def get_status(self):
            """Get current pool status information."""
            return {
                'current_pool_size': len(self.pool), 
                'max_pool_size': self.max_size, 
                'database_path': self.db_path,
                'connections_available': len(self.pool)
            }

        def __call__(self):
            """
            Get a connection with context manager support.

            Returns a DBConnection context manager for safe connection handling.
            """
            return DBConnection(self)


    #| export

    class DBConnection:
        """Context manager for safe database connection handling."""

        def __init__(self, pool: DBPool):  # DBPool instance to get/return connections from
            """Initialize connection context manager."""
            self.pool, self.conn = pool, None

        def __enter__(self):
            """Enter context manager and get database connection."""
            self.conn = self.pool.get_connection()
            return self.conn

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Exit context manager and return connection to pool."""
            if self.conn:
                self.pool.return_connection(self.conn)


    #| export

    def initialize_database_pool(max_size: int = 6):
        """Initialize the global database connection pool using global db_path."""
        global db_pool

        # Directly use the global db_path variable
        db_pool = DBPool(str(db_path), max_size)

        # Pre-populate pool with initial connections
        initial_connections = min(3, max_size)
        for _ in range(initial_connections): 
            try: 
                conn = db_pool.get_connection()
                db_pool.return_connection(conn)
            except Exception as e:
                print(f"Failed to pre-populate connection: {e}")
                break

        print(f"Initialized database pool: {db_pool.get_status()}")
    return (db_pool,)


@app.cell
def _(db_pool):
    # global variation makes sense everything needs access to the pool
    class BaseDAO:
        "Base Data Access Object with connection management using global pool"

        def with_conn(self, f):
            "Execute function with a connection from the global pool"
            with db_pool() as conn:
                return f(conn)
    return


@app.cell
def _(Input, Table, Tbody, Th, Thead, Tr, task_row):
    def task_table(tasks):
        return Table(
            Thead(
                Tr(
                    Th(Input(type="checkbox", data_bind_checked="$selections.length === $tasks.length")),
                    Th("Title"),
                    Th("Status"),
                    Th("Actions")
                )
            ),
            Tbody(
                *[task_row(task) for task in tasks]
            ),
            data_signals={"selections": [], "tasks": tasks}
        )
    return (task_table,)


@app.cell
def _(Button, Input, Td, Tr):
    def task_row(task):
        return Tr(
            Td(Input(type="checkbox", value=str(task['task_id']), 
                    data_bind_selections=f"task_{task['task_id']}")),
            Td(task['title']),
            Td("✓" if task['completed'] else "○"),
            Td(Button("Toggle", data_on_click=f"@put('/task/{task['task_id']}/toggle')"))
        )
    return (task_row,)


@app.cell
def _(FastHTML, T):
    fastapp = FastHTML(htmx=False, datastar=T)
    return (fastapp,)


@app.cell
def _(TaskDAO, fastapp, task_table):
    @fastapp.get('/tasks')
    def tasks_page(signals):
        task_dao = TaskDAO()
        tasks = task_dao.get_all_tasks()
        return task_table(tasks)
    return


@app.cell
def _(TaskDAO, fastapp, task_table):
    @fastapp.put('/task/{task_id}/toggle')
    def toggle_task(task_id: int, signals):
        task_dao = TaskDAO()
        # Get current status and flip it
        tasks = task_dao.get_all_tasks()
        current_task = next(t for t in tasks if t['task_id'] == task_id)
        new_status = not current_task['completed']

        task_dao.update_status(task_id, new_status)

        # Return updated table
        updated_tasks = task_dao.get_all_tasks()
        return task_table(updated_tasks)
    return


@app.cell
def _(TaskDAO, fastapp, task_table):

    @fastapp.put('/tasks/complete')
    def bulk_complete(signals):
        task_dao = TaskDAO()
        selected_ids = signals.get('selections', [])

        if selected_ids:
            task_dao.bulk_update_status(selected_ids, True)

        # Return updated table
        updated_tasks = task_dao.get_all_tasks()
        return task_table(updated_tasks)
    return


@app.cell
def _(TaskDAO, fastapp, task_table):

    @fastapp.put('/tasks/incomplete')
    def bulk_incomplete(signals):
        task_dao = TaskDAO()
        selected_ids = signals.get('selections', [])

        if selected_ids:
            task_dao.bulk_update_status(selected_ids, False)

        # Return updated table
        updated_tasks = task_dao.get_all_tasks()
        return task_table(updated_tasks)
    return


@app.cell
def _():
    # Simple fake database
    tasks = [
        {"task_id": 1, "title": "Test task 1", "completed": False},
        {"task_id": 2, "title": "Test task 2", "completed": True},
        {"task_id": 3, "title": "Test task 3", "completed": False}
    ]
    return (tasks,)


@app.cell
def _(fake_tasks, tasks):
    class TaskDAO:
        def get_all_tasks(self):
            return tasks

        def update_status(self, task_id, completed):
            for task in tasks:  
                if task['task_id'] == task_id:
                    task['completed'] = completed
                    return 1
            return 0

        def bulk_update_status(self, task_ids, completed):
            updated = 0
            for task in fake_tasks:
                if task['task_id'] in task_ids:
                    task['completed'] = completed
                    updated += 1
            return updated

        def create_task(self, title, description=""):
            new_id = max(task['task_id'] for task in fake_tasks) + 1
            new_task = {"task_id": new_id, "title": title, "description": description, "completed": False}
            fake_tasks.append(new_task)
            return new_id
    return (TaskDAO,)


@app.cell
def _():
    import os
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Jupyter Server (WIP)""")
    return


@app.cell
def _():
    # """Use FastHTML in Jupyter notebooks"""

    # # AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/api/06_jupyter.ipynb.

    # # %% auto 0


    # # %% ../nbs/api/06_jupyter.ipynb
    # import socket, time, uvicorn
    # from threading import Thread

    # from fastcore.meta import delegates
    # from fasthtml.common import show as _show
    # from fastcore.parallel import startthread
    # try: from IPython.display import HTML,Markdown,display
    # except ImportError: pass

    # # %% ../nbs/api/06_jupyter.ipynb
    # def nb_serve(app, log_level="error", port=8000, host='0.0.0.0', **kwargs):
    #     "Start a Jupyter compatible uvicorn server with ASGI `app` on `port` with `log_level`"
    #     server = uvicorn.Server(uvicorn.Config(app, log_level=log_level, host=host, port=port, **kwargs))
    #     async def async_run_server(server): await server.serve()
    #     @startthread
    #     def run_server(): asyncio.run(async_run_server(server))
    #     while not server.started: time.sleep(0.01)
    #     return server

    # # %% ../nbs/api/06_jupyter.ipynb
    # async def nb_serve_async(app, log_level="error", port=8000, host='0.0.0.0', **kwargs):
    #     "Async version of `nb_serve`"
    #     server = uvicorn.Server(uvicorn.Config(app, log_level=log_level, host=host, port=port, **kwargs))
    #     asyncio.get_running_loop().create_task(server.serve())
    #     while not server.started: await asyncio.sleep(0.01)
    #     return server

    # # %% ../nbs/api/06_jupyter.ipynb
    # def is_port_free(port, host='localhost'):
    #     "Check if `port` is free on `host`"
    #     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     try:
    #         sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #         sock.bind((host, port))
    #         return True
    #     except OSError: return False
    #     finally: sock.close()

    # # %% ../nbs/api/06_jupyter.ipynb
    # def wait_port_free(port, host='localhost', max_wait=3):
    #     "Wait for `port` to be free on `host`"
    #     start_time = time.time()
    #     while not is_port_free(port):
    #         if time.time() - start_time>max_wait: return print(f"Timeout")
    #         time.sleep(0.1)

    # # %% ../nbs/api/06_jupyter.ipynb
    # @delegates(_show)
    # def show(*s, **kwargs):
    #     "Same as fasthtml.components.show, but also adds `htmx.process()`"
    #     if IN_NOTEBOOK: return _show(*s, Script('if (window.htmx) htmx.process(document.body)'), **kwargs)
    #     return _show(*s, **kwargs)

    # # %% ../nbs/api/06_jupyter.ipynb
    # def render_ft():
    #     @patch
    #     def _repr_markdown_(self:FT): return to_xml(Div(self, Script('if (window.htmx) htmx.process(document.body)')))

    # # %% ../nbs/api/06_jupyter.ipynb
    # def htmx_config_port(port=8000):
    #     display(HTML('''
    # <script>
    # document.body.addEventListener('htmx:configRequest', (event) => {
    #     if(event.detail.path.includes('://')) return;
    #     htmx.config.selfRequestsOnly=false;
    #     event.detail.path = `${location.protocol}//${location.hostname}:%s${event.detail.path}`;
    # });
    # </script>''' % port))

    # # %% ../nbs/api/06_jupyter.ipynb
    # class JupyUvi:
    #     "Start and stop a Jupyter compatible uvicorn server with ASGI `app` on `port` with `log_level`"
    #     def __init__(self, app, log_level="error", host='0.0.0.0', port=8000, start=True, **kwargs):
    #         self.kwargs = kwargs
    #         store_attr(but='start')
    #         self.server = None
    #         if start: self.start()
    #         htmx_config_port(port)

    #     def start(self):
    #         self.server = nb_serve(self.app, log_level=self.log_level, host=self.host, port=self.port, **self.kwargs)

    #     async def start_async(self):
    #         self.server = await nb_serve_async(self.app, log_level=self.log_level, host=self.host, port=self.port, **self.kwargs)

    #     def stop(self):
    #         self.server.should_exit = True
    #         wait_port_free(self.port)

    # # %% ../nbs/api/06_jupyter.ipynb
    # class JupyUviAsync(JupyUvi):
    #     "Start and stop an async Jupyter compatible uvicorn server with ASGI `app` on `port` with `log_level`"
    #     def __init__(self, app, log_level="error", host='0.0.0.0', port=8000, **kwargs):
    #         super().__init__(app, log_level=log_level, host=host, port=port, start=False, **kwargs)

    #     async def start(self):
    #         self.server = await nb_serve_async(self.app, log_level=self.log_level, host=self.host, port=self.port, **self.kwargs)

    #     def stop(self):
    #         self.server.should_exit = True
    #         wait_port_free(self.port)

    # # %% ../nbs/api/06_jupyter.ipynb
    return


@app.cell
def _():
    # try:
    #     server = JupyUvi(fastapp, port=5010)
    # except NameError:
    #     server = nb_serve(fastapp, port=5010)
    return


@app.cell
def _():
    import socket
    import time
    from typing import Callable
    from fasthtml.jupyter import JupyUvi, is_port_free, wait_port_free
    return JupyUvi, is_port_free


@app.cell
def _(JupyUvi, Optional, is_port_free, mo):
    class MarimoServerController:
        """A marimo-compatible widget for controlling FastHTML nb_server instances"""
    
        def __init__(self, app, default_port: int = 8000, default_host: str = '0.0.0.0'):
            self.app = app
            self.default_port = default_port
            self.default_host = default_host
        
            # Create reactive state for server status
            self.get_server_running, self.set_server_running = mo.state(False)
            self.get_server_instance, self.set_server_instance = mo.state(None)
            self.get_status_message, self.set_status_message = mo.state("Server stopped")
        
            # Create UI elements
            self.port_input = mo.ui.number(
                value=default_port,
                start=1024,
                stop=65535,
                label="Port:",
                disabled=False
            )
        
            self.host_input = mo.ui.text(
                value=default_host,
                label="Host:",
                disabled=False
            )
        
            self.log_level_select = mo.ui.dropdown(
                options=["error", "warning", "info", "debug"],
                value="error",
                label="Log Level:"
            )
        
            # Start/Stop button with conditional logic
            self.control_button = mo.ui.button(
                label="Start Server",
                on_click=self._toggle_server,
                disabled=False
            )
        
            # Status display
            self.status_display = mo.ui.text(
                value="",
                label="Status:",
                disabled=True
            )
    
        def _toggle_server(self, _value=None):
            """Toggle server start/stop"""
            if self.get_server_running():
                self._stop_server()
            else:
                self._start_server()
    
        def _start_server(self):
            """Start the FastHTML server"""
            try:
                port = self.port_input.value
                host = self.host_input.value
                log_level = self.log_level_select.value
            
                # Check if port is available
                if not is_port_free(port, host):
                    self.set_status_message(f"Port {port} is already in use")
                    return
            
                # Create and start server
                server_instance = JupyUvi(
                    self.app,
                    port=port,
                    host=host,
                    log_level=log_level,
                    start=True
                )
            
                # Update state
                self.set_server_instance(server_instance)
                self.set_server_running(True)
                self.set_status_message(f"Server running on {host}:{port}")
            
                # Update button label
                self.control_button = mo.ui.button(
                    label="Stop Server",
                    on_click=self._toggle_server,
                    disabled=False
                )
            
            except Exception as e:
                self.set_status_message(f"Failed to start server: {str(e)}")
    
        def _stop_server(self):
            """Stop the FastHTML server"""
            try:
                server_instance = self.get_server_instance()
                if server_instance:
                    server_instance.stop()
                
                # Update state
                self.set_server_instance(None)
                self.set_server_running(False)
                self.set_status_message("Server stopped")
            
                # Update button label
                self.control_button = mo.ui.button(
                    label="Start Server",
                    on_click=self._toggle_server,
                    disabled=False
                )
            
            except Exception as e:
                self.set_status_message(f"Failed to stop server: {str(e)}")
    
        def get_server_url(self) -> Optional[str]:
            """Get the current server URL if running"""
            if self.get_server_running():
                host = self.host_input.value
                port = self.port_input.value
                return f"http://{host}:{port}"
            return None
    
        def render(self):
            """Render the complete widget UI"""
            # Create the main widget layout
            widget_content = mo.vstack([
                mo.md("## FastHTML Server Controller"),
            
                # Configuration inputs
                mo.hstack([self.port_input, self.host_input]),
                self.log_level_select,
            
                # Control button
                self.control_button,
            
                # Status display
                mo.md(f"**Status:** {self.get_status_message()}"),
            
                # Server URL (if running)
                mo.md(f"**URL:** {self.get_server_url() or 'Not running'}") if self.get_server_running() else mo.md(""),
            
                # Additional info
                mo.callout(
                    mo.md("""
    **Usage:**
    1. Configure the port and host settings
    2. Select the desired log level
    3. Click 'Start Server' to launch the FastHTML server
    4. Click 'Stop Server' to shut it down
                    """),
                    kind="info"
                )
            ])
        
            return widget_content


    # Example usage function
    def create_server_widget(app, port: int = 8000, host: str = '0.0.0.0'):
        """
        Create a server controller widget for a FastHTML app
    
        Args:
            app: FastHTML application instance
            port: Default port number
            host: Default host address
    
        Returns:
            MarimoServerController instance
        """
        return MarimoServerController(app, port, host)


    # Alternative simpler version using just marimo UI elements
    def simple_server_controller(app, port: int = 8000):
        """
        A simpler version using basic marimo UI elements
        """
        # State management
        get_running, set_running = mo.state(False)
        get_server, set_server = mo.state(None)
    
        # UI Elements
        port_input = mo.ui.number(value=port, start=1024, stop=65535, label="Port")
    
        def toggle_server(_value=None):
            if get_running():
                # Stop server
                server = get_server()
                if server:
                    server.stop()
                set_server(None)
                set_running(False)
            else:
                # Start server
                try:
                    server = JupyUvi(app, port=port_input.value, start=True)
                    set_server(server)
                    set_running(True)
                except Exception as e:
                    print(f"Error starting server: {e}")
    
        start_button = mo.ui.button(
            label="Stop Server" if get_running() else "Start Server",
            on_click=toggle_server
        )
    
        # Layout
        return mo.vstack([
            mo.md("### Server Control"),
            port_input,
            start_button,
            mo.md(f"Status: {'Running' if get_running() else 'Stopped'}"),
            mo.md(f"URL: http://localhost:{port_input.value}" if get_running() else "")
        ])
    return (create_server_widget,)


@app.cell
def _(create_server_widget, fastapp):
    # Create the server controller widget
    controller = create_server_widget(fastapp, port=5555)

    # Render the widget in your marimo notebook
    controller.render()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
