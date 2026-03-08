import os
import re
from jinja2 import Environment, FileSystemLoader
from schemas import WorkflowOptimise

def _sanitize_id(text: str) -> str:
    """Nettoie les identifiants pour qu'ils soient conformes aux variables Python et identifiants Airflow."""
    text = text.lower()
    return re.sub(r'[^a-z0-9_]', '_', text)

def generate_airflow_dag(workflow: WorkflowOptimise) -> str:
    """
    Parse le modèle WorkflowOptimise et génère un script DAG Airflow valide.
    """
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('airflow_dag.jinja2')

    tasks = []
    for noeud in workflow.noeuds:
        task_id = f"task_{_sanitize_id(noeud.id)}"

        # Mappage strict des noeuds Pydantic vers les opérateurs Airflow
        if noeud.type_noeud == "trigger":
            operator = "EmptyOperator"
        elif noeud.type_noeud == "automatique":
            operator = "PythonOperator"
        elif noeud.type_noeud == "humain":
            operator = "BashOperator"
        elif noeud.type_noeud == "decision":
            operator = "BranchPythonOperator"
        elif noeud.type_noeud == "fin":
            operator = "EmptyOperator"
        else:
            operator = "EmptyOperator"

        tasks.append({
            "task_id": task_id,
            "operator": operator,
            "label": noeud.label.replace("'", "\\'"),
            "description": noeud.description.replace("'", "\\'")
        })

    dependencies = []
    for lien in workflow.liens:
        source = f"task_{_sanitize_id(lien.de)}"
        target = f"task_{_sanitize_id(lien.vers)}"
        dependencies.append(f"{source} >> {target}")

    dag_id = f"dag_{_sanitize_id(workflow.titre_workflow)}"

    rendered_dag = template.render(
        dag_id=dag_id,
        description=workflow.description_transformation.replace("'", "\\'"),
        tasks=tasks,
        dependencies=dependencies
    )

    return rendered_dag
