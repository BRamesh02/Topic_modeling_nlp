"""
Mapping from raw `titulaire-soutien` / `titulaire-liste` labels to seven
political families. Rule order matters: more specific patterns first (so
'parti socialiste unifié' is caught as radical_left before the generic
'socialiste' rule sends it to socialist_left).
"""

import re

import pandas as pd


AFFILIATION_FIELDS = ["titulaire-soutien", "titulaire-liste"]


def compact(value):
    if pd.isna(value):
        return ""
    return str(value).lower()


def assign_party_family(row):
    parts = []
    for col in AFFILIATION_FIELDS:
        val = compact(row.get(col, ""))
        if val and "non mentionné" not in val:
            parts.append(val)
    labels = " ".join(parts)

    if not labels.strip():
        return "unclassified"

    if "front national" in labels:
        return "national_right"
    if "parti des forces nouvelles" in labels:
        return "national_right"

    if "lutte ouvrière" in labels or "lutte ouvriere" in labels:
        return "radical_left"
    if "ligue communiste" in labels:
        return "radical_left"
    if "parti socialiste unifié" in labels or "parti socialiste unifie" in labels:
        return "radical_left"
    if "parti ouvrier européen" in labels or "parti ouvrier europeen" in labels:
        return "radical_left"
    if "comités juquin" in labels or "comites juquin" in labels:
        return "radical_left"
    if "marxistes-léninistes" in labels or "marxistes-leninistes" in labels:
        return "radical_left"

    if "communiste" in labels:
        return "communist_left"

    if "écolog" in labels or "ecolog" in labels:
        return "ecologist"
    if "verts" in labels:
        return "ecologist"
    if "nature et animaux" in labels:
        return "ecologist"
    if "défense des animaux" in labels or "defense des animaux" in labels:
        return "ecologist"

    if "parti socialiste démocrate" in labels or "parti socialiste democrate" in labels:
        return "socialist_left"
    if "socialiste" in labels:
        return "socialist_left"
    if "radicaux de gauche" in labels or "radical de gauche" in labels:
        return "socialist_left"
    if "mouvement des citoyens" in labels:
        return "socialist_left"

    if "rassemblement pour la république" in labels or "rassemblement pour la republique" in labels:
        return "gaullist_right"
    if re.search(r"\brpr\b", labels):
        return "gaullist_right"
    if "union des démocrates pour la république" in labels or "union des democrates pour la republique" in labels:
        return "gaullist_right"
    if re.search(r"\budr\b", labels):
        return "gaullist_right"
    if "union des républicains de progrès" in labels or "union des republicains de progres" in labels:
        return "gaullist_right"
    if re.search(r"\burp\b", labels):
        return "gaullist_right"
    if "gaulliste" in labels or "gaullistes" in labels:
        return "gaullist_right"

    if "union pour la démocratie française" in labels or "union pour la democratie francaise" in labels:
        return "liberal_center_right"
    if re.search(r"\budf\b", labels):
        return "liberal_center_right"
    if "mouvement réformateur" in labels or "mouvement reformateur" in labels:
        return "liberal_center_right"
    if "centre démocratie et progrès" in labels or "centre democratie et progres" in labels:
        return "liberal_center_right"
    if "centre des démocrates sociaux" in labels or "centre des democrates sociaux" in labels:
        return "liberal_center_right"
    if re.search(r"\bcds\b", labels):
        return "liberal_center_right"
    if "centre démocrate" in labels or "centre democrate" in labels:
        return "liberal_center_right"
    if "républicains indépendants" in labels or "republicains independants" in labels:
        return "liberal_center_right"
    if "alliance républicaine" in labels or "alliance republicaine" in labels:
        return "liberal_center_right"
    if "centre national des indépendants" in labels or "centre national des independants" in labels:
        return "liberal_center_right"
    if re.search(r"\bcnip\b", labels):
        return "liberal_center_right"
    if "mouvement des démocrates" in labels or "mouvement des democrates" in labels:
        return "liberal_center_right"
    if "réformateur" in labels or "reformateur" in labels:
        return "liberal_center_right"
    if "parti républicain" in labels or "parti republicain" in labels:
        return "liberal_center_right"
    if "démocratie chrétienne" in labels or "democratie chretienne" in labels:
        return "liberal_center_right"
    if "parti radical" in labels:
        return "liberal_center_right"

    return "other"
