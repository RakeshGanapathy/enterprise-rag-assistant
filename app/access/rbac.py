"""
Role-Based Access Control for retrieval.

Two dimensions control what a user can retrieve:
  department    - which document groups the role can see
  access_level  - maximum sensitivity tier the role can see (numeric, inclusive)

Access levels (lower = less sensitive):
  0  public       anyone can read
  1  internal     employees only
  2  confidential managers and above
  3  restricted   named roles only (legal, exec, security)

Departments:
  hr          HR policies, benefits, compensation
  security    security policies, incident response
  product     product FAQs, release notes
  finance     financial reports, budgets
  all         meta-department: grants access across all departments

Role definitions map a role name to allowed departments and max access level.
Add new roles here without touching any other file.
"""
from __future__ import annotations

from dataclasses import dataclass

ACCESS_LEVELS = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}

# Reverse map: level int -> label
ACCESS_LEVEL_LABELS = {v: k for k, v in ACCESS_LEVELS.items()}


@dataclass(frozen=True)
class RolePolicy:
    departments: frozenset[str]   # which departments this role can read
    max_access_level: int         # maximum sensitivity tier (inclusive)


# Central role registry — edit this to add or change roles
ROLE_POLICIES: dict[str, RolePolicy] = {
    "admin": RolePolicy(
        departments=frozenset({"all"}),
        max_access_level=ACCESS_LEVELS["restricted"],
    ),
    "hr_manager": RolePolicy(
        departments=frozenset({"hr", "all"}),
        max_access_level=ACCESS_LEVELS["confidential"],
    ),
    "hr_staff": RolePolicy(
        departments=frozenset({"hr"}),
        max_access_level=ACCESS_LEVELS["internal"],
    ),
    "security_engineer": RolePolicy(
        departments=frozenset({"security", "all"}),
        max_access_level=ACCESS_LEVELS["restricted"],
    ),
    "support_agent": RolePolicy(
        departments=frozenset({"product"}),
        max_access_level=ACCESS_LEVELS["internal"],
    ),
    "employee": RolePolicy(
        departments=frozenset({"hr", "product", "all"}),
        max_access_level=ACCESS_LEVELS["public"],
    ),
    "anonymous": RolePolicy(
        departments=frozenset({"all"}),
        max_access_level=ACCESS_LEVELS["public"],
    ),
}

# Filename prefix -> department mapping used at ingest time
FILENAME_DEPARTMENT_MAP: dict[str, str] = {
    "hr_":        "hr",
    "security_":  "security",
    "incident_":  "security",
    "product_":   "product",
    "finance_":   "finance",
}

# Filename prefix -> access level mapping used at ingest time
FILENAME_ACCESS_LEVEL_MAP: dict[str, str] = {
    "hr_":        "confidential",
    "security_":  "restricted",
    "incident_":  "restricted",
    "product_":   "public",
    "finance_":   "confidential",
}


def get_role_policy(role: str) -> RolePolicy:
    """Return the policy for a role. Unknown roles get anonymous (public only)."""
    return ROLE_POLICIES.get(role, ROLE_POLICIES["anonymous"])


def infer_document_metadata(filename: str) -> dict[str, str]:
    """
    Infer department and access_level from a filename at ingest time.
    Falls back to department='general', access_level='internal' when unknown.
    """
    lower = filename.lower()
    department = "general"
    access_level = "internal"

    for prefix, dept in FILENAME_DEPARTMENT_MAP.items():
        if lower.startswith(prefix):
            department = dept
            break

    for prefix, level in FILENAME_ACCESS_LEVEL_MAP.items():
        if lower.startswith(prefix):
            access_level = level
            break

    return {"department": department, "access_level": access_level}
