import graphviz
from schemas import WorkflowOptimise

def render_workflow(workflow: WorkflowOptimise) -> bytes:
    """
    Génère une représentation visuelle (SVG) du workflow optimisé
    à partir du modèle Pydantic strict.
    """
    # Configuration de l'architecture du graphe
    dot = graphviz.Digraph(
        comment='Workflow Optimisé',
        format='svg',
        graph_attr={
            'rankdir': 'TB',  # Top to Bottom
            'fontname': 'Helvetica',
            'nodesep': '0.5',
            'ranksep': '0.8',
            'bgcolor': 'transparent'
        },
        node_attr={
            'fontname': 'Helvetica',
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': '#F8FAFC',
            'color': '#CBD5E1',
            'fontcolor': '#1E293B',
            'penwidth': '2'
        },
        edge_attr={
            'fontname': 'Helvetica',
            'color': '#94A3B8',
            'fontsize': '10',
            'penwidth': '1.5'
        }
    )

    # Dictionnaire de mapping pour le typage visuel des noeuds
    type_colors = {
        'humain': {'fillcolor': '#EFF6FF', 'color': '#3B82F6'},      # Bleu
        'ia': {'fillcolor': '#F5F3FF', 'color': '#8B5CF6'},          # Violet
        'automatique': {'fillcolor': '#ECFDF5', 'color': '#10B981'}, # Vert
        'decision': {'fillcolor': '#FFFBEB', 'color': '#F59E0B', 'shape': 'diamond'} # Jaune
    }

    # 1. Génération des noeuds à partir de l'objet Pydantic
    for noeud in workflow.noeuds:
        attrs = type_colors.get(noeud.type_noeud.lower(), {})
        label = f"<<B>{noeud.label}</B><BR/>"
        label += f"<FONT POINT-SIZE='10' COLOR='#64748B'>{noeud.type_noeud.upper()}</FONT>>"

        dot.node(
            str(noeud.id),
            label=label,
            shape=attrs.get('shape', 'box'),
            fillcolor=attrs.get('fillcolor', '#F8FAFC'),
            color=attrs.get('color', '#CBD5E1')
        )

    # 2. Génération des arêtes (liens) à partir de l'objet Pydantic
    for lien in workflow.liens:
        edge_attrs = {}
        if getattr(lien, 'condition', None):
            edge_attrs['label'] = f" {lien.condition} "
            edge_attrs['fontcolor'] = '#475569'

        dot.edge(str(lien.de), str(lien.vers), **edge_attrs)

    return dot.pipe()
