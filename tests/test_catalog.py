"""
tests/test_catalog.py — test dello Step 1 (collasso.catalog), secondo il
piano vincolante in STATUS.md ("Piano ciclo corrente (Step 1) — REV. 2"),
sezione 5 (5.1-5.8).
"""

from __future__ import annotations

import pytest

from collasso.catalog import (
    MASSA_NUCLEO_MAX_MSUN,
    MASSA_NUCLEO_MIN_MSUN,
    N_POLITROPICO_MAX,
    N_POLITROPICO_MIN,
    YE_MAX,
    YE_MIN,
    _REFERENCE_CATALOG_PATH,
    Progenitor,
    get_progenitor_by_id,
    load_catalog_from_csv,
    load_reference_catalog,
)


# --- Fixture di comodo -------------------------------------------------

@pytest.fixture(scope="module")
def catalogo_riferimento():
    return load_reference_catalog()


def _progenitor_valido(**overrides) -> dict[str, float | str]:
    """Dizionario di parametri validi di base per costruire un Progenitor,
    con possibilita' di sovrascrivere singoli campi nei test negativi (5.5).
    """
    base = dict(
        id="test",
        massa_zams_msun=20.0,
        massa_nucleo_msun=1.5,
        densita_centrale_gcm3=7.5e9,
        raggio_iniziale_km=1900.0,
        ye=0.46,
        n_politropico=2.0,
        fonte="fonte di test -- placeholder",
    )
    base.update(overrides)
    return base


# --- 5.1: caricamento catalogo di riferimento ---------------------------
# Aggiornato nel ciclo "Espansione catalogo + Betelgeuse": 6 progenitori
# reali (s15/s20/s25/s30/s35/s40, tutti dalla Tabella 1 di O'Connor & Ott
# 2011) piu' 1 voce proxy dichiarata ("betelgeuse", riusa s20WHW02).

def test_5_1_caricamento_catalogo_riferimento(catalogo_riferimento):
    assert len(catalogo_riferimento) == 7
    ids = {p.id for p in catalogo_riferimento}
    assert ids == {"s15", "s20", "s25", "s30", "s35", "s40", "betelgeuse"}


# --- 5.2: range fisico Ye sui valori concreti del catalogo ---------------

def test_5_2_range_ye_valori_concreti(catalogo_riferimento):
    for p in catalogo_riferimento:
        assert YE_MIN <= p.ye <= YE_MAX, (
            f"{p.id}: ye={p.ye} fuori dal range [{YE_MIN}, {YE_MAX}]"
        )


# --- 5.3: range fisico n sui valori concreti del catalogo ---------------

def test_5_3_range_n_politropico_valori_concreti(catalogo_riferimento):
    for p in catalogo_riferimento:
        assert N_POLITROPICO_MIN <= p.n_politropico <= N_POLITROPICO_MAX, (
            f"{p.id}: n_politropico={p.n_politropico} fuori dal range "
            f"[{N_POLITROPICO_MIN}, {N_POLITROPICO_MAX}]"
        )


# --- 5.4: masse positive, nucleo <= zams, criterio Chandrasekhar --------

def test_5_4_masse_positive_e_coerenti(catalogo_riferimento):
    for p in catalogo_riferimento:
        assert p.massa_zams_msun > 0
        assert p.massa_nucleo_msun > 0
        assert p.massa_nucleo_msun <= p.massa_zams_msun


def test_5_4_criterio_plausibilita_chandrasekhar(catalogo_riferimento):
    for p in catalogo_riferimento:
        assert MASSA_NUCLEO_MIN_MSUN <= p.massa_nucleo_msun <= MASSA_NUCLEO_MAX_MSUN, (
            f"{p.id}: massa_nucleo_msun={p.massa_nucleo_msun} fuori dal "
            f"range Chandrasekhar [{MASSA_NUCLEO_MIN_MSUN}, "
            f"{MASSA_NUCLEO_MAX_MSUN}]"
        )


# --- 5.5: validazione dataclass, casi negativi ai limiti dei range ------

@pytest.mark.parametrize("ye_non_valido", [0.6, 0.3])
def test_5_5_ye_fuori_range_solleva_errore(ye_non_valido):
    with pytest.raises(ValueError):
        Progenitor(**_progenitor_valido(ye=ye_non_valido))


@pytest.mark.parametrize("n_non_valido", [0.5, 4.0])
def test_5_5_n_politropico_fuori_range_solleva_errore(n_non_valido):
    with pytest.raises(ValueError):
        Progenitor(**_progenitor_valido(n_politropico=n_non_valido))


@pytest.mark.parametrize("massa_nucleo_non_valida", [0.8, 3.0])
def test_5_5_massa_nucleo_fuori_range_chandrasekhar_solleva_errore(massa_nucleo_non_valida):
    with pytest.raises(ValueError):
        Progenitor(**_progenitor_valido(massa_nucleo_msun=massa_nucleo_non_valida))


def test_5_5_progenitor_valido_non_solleva_errore():
    # Caso di controllo positivo: nessuna eccezione con parametri validi.
    p = Progenitor(**_progenitor_valido())
    assert p.id == "test"


# --- 5.6: get_progenitor_by_id ------------------------------------------

def test_5_6_get_progenitor_by_id_valido(catalogo_riferimento):
    p = get_progenitor_by_id(catalogo_riferimento, "s20")
    assert p.id == "s20"
    assert p.massa_zams_msun == 20.0


def test_5_6_get_progenitor_by_id_inesistente_solleva_errore(catalogo_riferimento):
    with pytest.raises(ValueError):
        get_progenitor_by_id(catalogo_riferimento, "s99")


# --- 5.7: tracciabilita' della provenienza per campo ---------------------
# Aggiornato nel ciclo "Integrazione catalogo reale" (successivo a Step 8):
# il catalogo di riferimento non e' piu' interamente placeholder — massa
# del nucleo e ye sono dati di letteratura (O'Connor & Ott 2011), mentre
# n_politropico/raggio_iniziale_km restano derivati internamente. Il campo
# `fonte` deve tracciare esplicitamente questa distinzione per ogni stella.

def test_5_7_fonte_traccia_letteratura_e_derivazione_interna(catalogo_riferimento):
    for p in catalogo_riferimento:
        fonte_lower = p.fonte.lower()
        assert "o'connor" in fonte_lower or "oconnor" in fonte_lower, (
            f"{p.id}: il campo 'fonte' non cita la fonte di letteratura "
            "(O'Connor & Ott 2011) per massa_nucleo_msun/ye"
        )
        assert "derivat" in fonte_lower, (
            f"{p.id}: il campo 'fonte' non dichiara esplicitamente che "
            "n_politropico/raggio_iniziale_km sono derivati internamente"
        )


# --- nota_proxy: solo "betelgeuse" e' una voce proxy dichiarata ----------
# (ciclo "Espansione catalogo + Betelgeuse").

def test_nota_proxy_solo_betelgeuse(catalogo_riferimento):
    for p in catalogo_riferimento:
        if p.id == "betelgeuse":
            assert p.nota_proxy != "", "betelgeuse deve avere nota_proxy non vuota"
            assert "proxy" in p.nota_proxy.lower()
        else:
            assert p.nota_proxy == "", f"{p.id}: nota_proxy dovrebbe essere vuota (non e' una voce proxy)"


def test_betelgeuse_riusa_esattamente_i_valori_di_s20(catalogo_riferimento):
    betelgeuse = get_progenitor_by_id(catalogo_riferimento, "betelgeuse")
    s20 = get_progenitor_by_id(catalogo_riferimento, "s20")
    assert betelgeuse.massa_nucleo_msun == s20.massa_nucleo_msun
    assert betelgeuse.ye == s20.ye
    assert betelgeuse.densita_centrale_gcm3 == s20.densita_centrale_gcm3
    assert betelgeuse.n_politropico == s20.n_politropico
    assert betelgeuse.raggio_iniziale_km == s20.raggio_iniziale_km


# --- 5.8: parsing CSV con commento in testa -------------------------------

def test_5_8_parsing_csv_con_commento_in_testa():
    catalogo = load_catalog_from_csv(_REFERENCE_CATALOG_PATH)
    assert len(catalogo) == 7
    # Nessuna riga di commento deve essere stata interpretata come dato
    # (incluse quelle interposte prima della riga "betelgeuse", non solo
    # quelle in testa al file): tutti gli id devono essere quelli attesi,
    # non frammenti di commento.
    assert {p.id for p in catalogo} == {"s15", "s20", "s25", "s30", "s35", "s40", "betelgeuse"}
    for p in catalogo:
        assert isinstance(p, Progenitor)
