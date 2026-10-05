"""Map EPUB reading-order documents to the book's top-level TOC chapters."""

import copy
import posixpath
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup, NavigableString


def _toc_entries(nodes):
    for node in nodes:
        if isinstance(node, tuple):
            parent, children = node
            if getattr(parent, 'href', None):
                yield parent
            else:
                yield from _toc_entries(children)
        elif getattr(node, 'href', None):
            yield node


def _document_slice(doc, soup, positions, start, end):
    """Retain ancestors so slicing at an anchor preserves valid HTML structure."""
    def select(node):
        position = positions[id(node)]
        if isinstance(node, NavigableString):
            return copy.copy(node) if start <= position < end else None
        children = [selected for child in node.children
                    if (selected := select(child)) is not None]
        if not children and not start <= position < end:
            return None
        selected = soup.new_tag(node.name, attrs=copy.deepcopy(node.attrs))
        for child in children:
            selected.append(child)
        return selected

    body = select(soup.body)
    segment = copy.copy(doc)
    segment.content = str(body).encode('utf-8')
    return segment


def chapter_documents(book):
    """Yield (document segment, chapter metadata) in spine order.

    Nested TOC entries remain in their parent chapter. Without a TOC, each
    spine document is a separate chapter. Content before the TOC is retained.
    """
    documents = []
    for item in book.spine:
        item_id = item[0] if isinstance(item, tuple) else item
        doc = item_id if hasattr(item_id, 'get_body_content') else book.get_item_with_id(item_id)
        if doc is not None and hasattr(doc, 'get_body_content'):
            documents.append(doc)
    names = {posixpath.normpath(unquote(doc.file_name)) for doc in documents}
    targets = {}
    for entry in _toc_entries(book.toc):
        href = urlsplit(entry.href)
        name = posixpath.normpath(unquote(href.path))
        if href.scheme or name not in names:
            continue
        anchor = unquote(href.fragment)
        targets.setdefault(name, []).append((anchor, {
            'key': f'{name}#{anchor}', 'title': str(entry.title),
        }))

    current = {'key': '__frontmatter__', 'title': 'Front matter'}
    for doc in documents:
        name = posixpath.normpath(unquote(doc.file_name))
        soup = BeautifulSoup(doc.get_content(), 'html.parser')
        if not targets:
            heading = soup.find(['h1', 'h2', 'h3', 'title'])
            yield doc, {'key': name, 'title': heading.get_text(' ', strip=True) if heading else name}
            continue
        boundaries = targets.get(name, [])
        if not boundaries:
            yield doc, dict(current)
            continue
        if soup.body is None:
            raise ValueError(f'Cannot locate chapter boundaries in {name}: no body')
        nodes = [soup.body, *soup.body.descendants]
        positions = {id(node): index for index, node in enumerate(nodes)}
        starts = {}
        for anchor, chapter in boundaries:
            element = soup.body.find(id=anchor) if anchor else soup.body
            if element is None:
                element = soup.body.find(attrs={'name': anchor})
            # Some EPUBs contain stale TOC fragments after splitting files.
            # A sole top-level target still identifies the whole document.
            if element is None and len(boundaries) == 1:
                element = soup.body
            if element is None:
                raise ValueError(f'Cannot locate TOC chapter anchor {name}#{anchor}')
            starts.setdefault(positions[id(element)], chapter)
        ordered = sorted(starts.items())
        # Most Calibre documents have just one chapter boundary. Keep the
        # original document intact in that case, including leading whitespace.
        if len(ordered) == 1:
            position, chapter = ordered[0]
            prefix_text = ''.join(str(node) for node in nodes[:position]
                                  if isinstance(node, NavigableString)).strip()
            if not prefix_text:
                current = chapter
                yield doc, dict(current)
                continue
        cursor = 0
        for position, chapter in ordered:
            if position > cursor:
                yield _document_slice(doc, soup, positions, cursor, position), dict(current)
            current = chapter
            cursor = position
        yield _document_slice(doc, soup, positions, cursor, len(nodes)), dict(current)


def chapter_groups(blocks, original_blocks):
    """Group kept block indices, using original IDs to survive editor changes."""
    by_id = {block['id']: block.get('book_chapter') for block in original_blocks}
    groups = []
    previous_key = None
    for index, block in enumerate(blocks):
        if not (block['keep'] and block['text'].strip()):
            continue
        chapter = by_id.get(block['id'])
        if not chapter:
            raise ValueError('Book chapter mapping is missing; reload the ebook before exporting.')
        if not groups or chapter['key'] != previous_key:
            groups.append({'title': chapter['title'], 'indices': []})
        groups[-1]['indices'].append(index)
        previous_key = chapter['key']
    return groups
