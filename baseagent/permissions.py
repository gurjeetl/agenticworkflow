from langchain_core.tools import BaseTool


def filter_tools_by_permission(
    tools: list[BaseTool],
    user_roles: list[str] | None = None,
) -> list[BaseTool]:
    """Return the subset of *tools* the user is allowed to call.

    Override per agent if role-based access control is needed.
    The default passes all tools through unchanged.
    """
    return tools
