from lxml import etree
from mdit_py_plugins.dollarmath import dollarmath_plugin


def update_mdit(mdit):
    mdit.use(dollarmath_plugin)


def _render_math_inline(node, context):
    return f"$${node.content}$$"


def _render_math_block(node, context):
    return f"\n$${node.content}$$\n"


def _render_html_inline(node, context):
    return _process_html(node.content)


def _render_html_block(node, context):
    return _process_html(node.content.lstrip())


def _process_html(html):
    xml = etree.XML(html)
    if xml.tag == 'markdown':
        return xml.text.strip()
    else:
        return html


RENDERERS = {
    "html_block": _render_html_block,
    "html_inline": _render_html_inline,
    "math_block": _render_math_block,
    "math_inline": _render_math_inline,
}
