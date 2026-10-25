"""HTTP route modules (decision F-4, DOCBOT-1800).

Convention for every NEW route (existing routes in ``api/index.py`` stay put
until they are deliberately migrated):

* One module per area: ``api/routes/<area>_routes.py`` (e.g.
  ``billing_routes.py``, ``usage_routes.py``, ``onboarding_routes.py``).
* Each module exposes a module-level ``router = APIRouter(prefix=..., tags=[...])``.
  Handlers stay thin: authz dependency -> service call -> Pydantic response.
* Request/response models live in a models module (``api/<area>_models.py``,
  e.g. ``api/billing_models.py``) with a Zod mirror in
  ``src/lib/schemas/<area>.ts``. No inline ``BaseModel`` classes in route files.
* Business logic lives in ``api/<area>_service.py`` / ``api/utils/``.
* ``api/index.py`` only does ``app.include_router(<area>_routes.router)``.
  That wiring is owned by the security lane so route guards are reviewed in
  one place; every route must carry a guard or be listed in ``PUBLIC_ROUTES``
  (DOCBOT-1625 route-guard sweep).

Keeping routes out of ``index.py`` lets parallel lanes add endpoints without
merge conflicts in a single ~3,000-line file.
"""
