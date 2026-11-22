"""Map HM Land Registry county names to English regions and Wales.

The county field is the ceremonial or unitary county recorded at registration,
not a statistical region. Each name below is assigned to the region that
contains that county. Postcode areas are not used for this mapping because
several of them cross regional boundaries.
"""

from __future__ import annotations

REGIONS = (
    "North East",
    "North West",
    "Yorkshire and The Humber",
    "East Midlands",
    "West Midlands",
    "East of England",
    "London",
    "South East",
    "South West",
    "Wales",
)

_COUNTY_TO_REGION: dict[str, str] = {
    # North East
    "COUNTY DURHAM": "North East",
    "DARLINGTON": "North East",
    "HARTLEPOOL": "North East",
    "MIDDLESBROUGH": "North East",
    "NORTHUMBERLAND": "North East",
    "REDCAR AND CLEVELAND": "North East",
    "STOCKTON-ON-TEES": "North East",
    "TYNE AND WEAR": "North East",
    # North West
    "BLACKBURN WITH DARWEN": "North West",
    "BLACKPOOL": "North West",
    "CHESHIRE EAST": "North West",
    "CHESHIRE WEST AND CHESTER": "North West",
    "CUMBERLAND": "North West",
    "GREATER MANCHESTER": "North West",
    "HALTON": "North West",
    "LANCASHIRE": "North West",
    "MERSEYSIDE": "North West",
    "WARRINGTON": "North West",
    "WESTMORLAND AND FURNESS": "North West",
    # Yorkshire and The Humber
    "CITY OF KINGSTON UPON HULL": "Yorkshire and The Humber",
    "EAST RIDING OF YORKSHIRE": "Yorkshire and The Humber",
    "NORTH EAST LINCOLNSHIRE": "Yorkshire and The Humber",
    "NORTH LINCOLNSHIRE": "Yorkshire and The Humber",
    "NORTH YORKSHIRE": "Yorkshire and The Humber",
    "SOUTH YORKSHIRE": "Yorkshire and The Humber",
    "WEST YORKSHIRE": "Yorkshire and The Humber",
    "YORK": "Yorkshire and The Humber",
    # East Midlands
    "CITY OF DERBY": "East Midlands",
    "CITY OF NOTTINGHAM": "East Midlands",
    "DERBYSHIRE": "East Midlands",
    "LEICESTER": "East Midlands",
    "LEICESTERSHIRE": "East Midlands",
    "LINCOLNSHIRE": "East Midlands",
    "NORTH NORTHAMPTONSHIRE": "East Midlands",
    "NOTTINGHAMSHIRE": "East Midlands",
    "RUTLAND": "East Midlands",
    "WEST NORTHAMPTONSHIRE": "East Midlands",
    # West Midlands
    "HEREFORDSHIRE": "West Midlands",
    "SHROPSHIRE": "West Midlands",
    "STAFFORDSHIRE": "West Midlands",
    "STOKE-ON-TRENT": "West Midlands",
    "WARWICKSHIRE": "West Midlands",
    "WEST MIDLANDS": "West Midlands",
    "WORCESTERSHIRE": "West Midlands",
    "WREKIN": "West Midlands",
    # East of England
    "BEDFORD": "East of England",
    "CAMBRIDGESHIRE": "East of England",
    "CENTRAL BEDFORDSHIRE": "East of England",
    "CITY OF PETERBOROUGH": "East of England",
    "ESSEX": "East of England",
    "HERTFORDSHIRE": "East of England",
    "LUTON": "East of England",
    "NORFOLK": "East of England",
    "SOUTHEND-ON-SEA": "East of England",
    "SUFFOLK": "East of England",
    "THURROCK": "East of England",
    # London
    "GREATER LONDON": "London",
    # South East
    "BRACKNELL FOREST": "South East",
    "BRIGHTON AND HOVE": "South East",
    "BUCKINGHAMSHIRE": "South East",
    "EAST SUSSEX": "South East",
    "HAMPSHIRE": "South East",
    "ISLE OF WIGHT": "South East",
    "KENT": "South East",
    "MEDWAY": "South East",
    "MILTON KEYNES": "South East",
    "OXFORDSHIRE": "South East",
    "PORTSMOUTH": "South East",
    "READING": "South East",
    "SLOUGH": "South East",
    "SOUTHAMPTON": "South East",
    "SURREY": "South East",
    "WEST BERKSHIRE": "South East",
    "WEST SUSSEX": "South East",
    "WINDSOR AND MAIDENHEAD": "South East",
    "WOKINGHAM": "South East",
    # South West
    "BATH AND NORTH EAST SOMERSET": "South West",
    "BOURNEMOUTH, CHRISTCHURCH AND POOLE": "South West",
    "CITY OF BRISTOL": "South West",
    "CITY OF PLYMOUTH": "South West",
    "CORNWALL": "South West",
    "DEVON": "South West",
    "DORSET": "South West",
    "GLOUCESTERSHIRE": "South West",
    "ISLES OF SCILLY": "South West",
    "NORTH SOMERSET": "South West",
    "SOMERSET": "South West",
    "SOUTH GLOUCESTERSHIRE": "South West",
    "SWINDON": "South West",
    "TORBAY": "South West",
    "WILTSHIRE": "South West",
    # Wales
    "BLAENAU GWENT": "Wales",
    "BRIDGEND": "Wales",
    "CAERPHILLY": "Wales",
    "CARDIFF": "Wales",
    "CARMARTHENSHIRE": "Wales",
    "CEREDIGION": "Wales",
    "CONWY": "Wales",
    "DENBIGHSHIRE": "Wales",
    "FLINTSHIRE": "Wales",
    "GWYNEDD": "Wales",
    "ISLE OF ANGLESEY": "Wales",
    "MERTHYR TYDFIL": "Wales",
    "MONMOUTHSHIRE": "Wales",
    "NEATH PORT TALBOT": "Wales",
    "NEWPORT": "Wales",
    "PEMBROKESHIRE": "Wales",
    "POWYS": "Wales",
    "RHONDDA CYNON TAFF": "Wales",
    "SWANSEA": "Wales",
    "THE VALE OF GLAMORGAN": "Wales",
    "TORFAEN": "Wales",
    "WREXHAM": "Wales",
}


def county_to_region(county: str | None) -> str | None:
    """Return the region for a Land Registry county name, or None if unknown."""
    if county is None:
        return None
    key = " ".join(str(county).strip().upper().split())
    if not key:
        return None
    return _COUNTY_TO_REGION.get(key)


def known_counties() -> frozenset[str]:
    """County names this mapping recognises, in Land Registry's uppercase form."""
    return frozenset(_COUNTY_TO_REGION)
