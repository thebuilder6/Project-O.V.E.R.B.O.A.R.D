"""Visualization tools, trajectory plotting, field geometry, and web studio assets."""

__all__ = [
    "Collision",
    "FieldModel",
    "Obstacle",
    "field_from_dict",
    "list_fields",
    "load_all_fields",
    "load_field",
    "autocrop",
    "compose_mat",
    "render_pdf_page",
]


def __getattr__(name: str):
    if name in (
        "Collision",
        "FieldModel",
        "Obstacle",
        "field_from_dict",
        "list_fields",
        "load_all_fields",
        "load_field",
    ):
        from overboard.viz import fields
        return getattr(fields, name)
    if name in ("autocrop", "compose_mat", "render_pdf_page"):
        from overboard.viz import mat_import
        return getattr(mat_import, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
