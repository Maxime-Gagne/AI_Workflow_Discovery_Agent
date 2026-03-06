import re
import graphviz
from schemas import WorkflowOptimise

def _sanitize(text: str) -> str:
    """
    Supprime ou remplace les caractères qui cassent le parser XML de Graphviz.
    Les labels HTML de Graphviz ne supportent pas les entités Unicode non-ASCII
    directement — on utilise des labels plain text à la place.
    """
    if not text:
        return ""
    # Tronquer les labels trop longs
    if len(text) > 40:
        text = text[:38] + ".."
    return text

def render_workflow(workflow: WorkflowOptimise) -> bytes:
    """
    Génère une représentation visuelle (SVG) du workflow optimisé
    à partir du modèle Pydantic strict.
    Utilise des labels plain text (pas HTML) pour éviter les erreurs
    d'encodage XML avec les caractères accentués.
    """
    dot = graphviz.Digraph(
        comment='Workflow Optimise',
        format='svg',
        graph_attr={
            'rankdir': 'TB',
            'fontname': 'Helvetica',
            'nodesep': '0.5',
            'ranksep': '0.8',
            'bgcolor': 'transparent',
            'charset': 'utf-8',
        },
        node_attr={
            'fontname': 'Helvetica',
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': '#F8FAFC',
            'color': '#CBD5E1',
            'fontcolor': '#1E293B',
            'penwidth': '2',
        },
        edge_attr={
            'fontname': 'Helvetica',
            'color': '#94A3B8',
            'fontsize': '10',
            'penwidth': '1.5',
        }
    )

    type_colors = {
        'trigger':     {'fillcolor': '#FEF3C7', 'color': '#F59E0B'},
        'humain':      {'fillcolor': '#EFF6FF', 'color': '#3B82F6'},
        'ia':          {'fillcolor': '#F5F3FF', 'color': '#8B5CF6'},
        'automatique': {'fillcolor': '#ECFDF5', 'color': '#10B981'},
        'decision':    {'fillcolor': '#FFFBEB', 'color': '#F59E0B', 'shape': 'diamond'},
        'fin':         {'fillcolor': '#F1F5F9', 'color': '#64748B'},
    }

    for noeud in workflow.noeuds:
        ntype = noeud.type_noeud.lower()
        attrs = type_colors.get(ntype, {})

        # Label plain text sur 2 lignes : nom + type
        # \n fonctionne en plain text, contrairement aux labels HTML avec accents
        label = f"{_sanitize(noeud.label)}\n[{noeud.type_noeud.upper()}]"

        dot.node(
            str(noeud.id),
            label=label,
            shape=attrs.get('shape', 'box'),
            fillcolor=attrs.get('fillcolor', '#F8FAFC'),
            color=attrs.get('color', '#CBD5E1'),
        )

    for lien in workflow.liens:
        edge_attrs = {}
        condition = getattr(lien, 'condition', None)
        if condition:
            edge_attrs['label'] = f" {_sanitize(condition)} "
            edge_attrs['fontcolor'] = '#475569'

        dot.edge(str(lien.de), str(lien.vers), **edge_attrs)

    return dot.pipe()
