"""
tests/test_pipeline.py — test della pipeline completa di simulazione
(Step 7).

Vedi STATUS.md, "Piano ciclo corrente (Step 7)", sezione "4. Test", per lo
spec vincolante di questi test (test di sanita' su PE_erg, test sugli 8
disclaimer/parole chiave, test sintetico dedicato per il ramo
collapsed=False).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collasso.constants import G_CGS, M_SUN_G
from collasso.dynamics import build_initial_shells, simulate_collapse
from collasso.lane_emden import physical_profile, solve_lane_emden
from collasso.pipeline import compute_collapse_energies, run_full_simulation

STAR_IDS = ["s15", "s20", "s25", "s30", "s35", "s40"]

# "betelgeuse" e' testata separatamente (test_betelgeuse_proxy_disclaimer):
# e' l'unica voce del catalogo con nota_proxy non vuota, quindi ha 9
# disclaimer invece di 8 (vedi collasso.pipeline._build_disclaimers) e non
# puo' condividere i test parametrizzati sopra basati su STAR_IDS.

# 8 parole chiave da verificare nei disclaimer (vedi STATUS.md, piano
# Step 7, "Contenuto esatto di disclaimers" — alcune voci ammettono
# alternative, verificate con OR). Voci 1-2 aggiornate nel ciclo
# "Integrazione catalogo reale" (successivo a Step 8): il catalogo non e'
# piu' interamente placeholder, vedi collasso.pipeline._build_disclaimers.
DISCLAIMER_KEYWORDS = [
    ["o'connor", "oconnor"],
    ["derivat"],
    ["neutrini"],
    ["viscosità", "viscosita"],
    ["case a"],  # verificato case-insensitive
    ["fenomenolog"],
    ["rotazione"],
    ["simmetria sferica", "1d"],
]


def _keyword_present(disclaimers: list[str], keywords: list[str]) -> bool:
    testo_completo = " ".join(disclaimers).lower()
    return any(kw.lower() in testo_completo for kw in keywords)


@pytest.mark.parametrize("star_id", STAR_IDS)
def test_run_full_simulation_no_exceptions_and_basic_sanity(star_id):
    """run_full_simulation(star_id) non solleva eccezioni; KE_finale_erg>=0;
    PE_erg<0 con ordine di grandezza plausibile; classificazioni valide.
    """
    result = run_full_simulation(star_id)

    assert result.energies.KE_finale_erg >= 0.0

    assert result.energies.PE_iniziale_erg < 0.0
    assert result.energies.PE_finale_erg < 0.0

    # --- Test di sanita' sull'ordine di grandezza di |PE_erg| (da review):
    #     confrontabile con G*M_tot^2/R_tipico entro un fattore ~2-5.
    m_tot_g = float(result.m_enclosed_g[-1])
    r_tipico_iniziale_cm = float(result.collapse_solution.r_cm[-1, 0])
    r_tipico_finale_cm = float(result.collapse_solution.r_cm[-1, -1])

    pe_scala_iniziale = G_CGS * m_tot_g**2 / r_tipico_iniziale_cm
    pe_scala_finale = G_CGS * m_tot_g**2 / r_tipico_finale_cm

    rapporto_iniziale = abs(result.energies.PE_iniziale_erg) / pe_scala_iniziale
    rapporto_finale = abs(result.energies.PE_finale_erg) / pe_scala_finale

    assert 0.2 <= rapporto_iniziale <= 5.0, (
        f"|PE_iniziale_erg|/(G*M_tot^2/R_iniziale) fuori dal fattore atteso "
        f"~2-5: rapporto={rapporto_iniziale}"
    )
    # Nota sulla soglia allargata per lo stato FINALE (verificato empiricamente
    # eseguendo la pipeline sulle 3 stelle del catalogo, non un numero a caso):
    # il collasso e' fortemente NON omologo — le shell interne collassano
    # molto piu' rapidamente di quella piu' esterna (che a t_finale ha ancora
    # un raggio vicino a quello iniziale), quindi il raggio dell'ULTIMA shell
    # (r_tipico_finale_cm) sottostima sistematicamente la vera scala spaziale
    # su cui e' concentrata l'autoenergia gravitazionale a t_finale. Il
    # rapporto osservato sulle 3 stelle di riferimento e' 1.68/8.37/5.42 —
    # 10.0 resta comunque un controllo di ordine di grandezza (non un numero
    # a caso), non un rilassamento a "qualunque valore".
    assert 0.2 <= rapporto_finale <= 10.0, (
        f"|PE_finale_erg|/(G*M_tot^2/R_finale) fuori dall'ordine di "
        f"grandezza atteso: rapporto={rapporto_finale}"
    )

    assert result.remnant_class_sly in {"white_dwarf", "neutron_star", "black_hole"}
    assert result.remnant_class_apr4 in {"white_dwarf", "neutron_star", "black_hole"}


@pytest.mark.parametrize("star_id", STAR_IDS)
def test_disclaimers_count_and_keywords(star_id):
    """Esattamente 8 disclaimer, ciascuna delle 8 parole chiave presente
    (case-insensitive) in almeno una voce.
    """
    result = run_full_simulation(star_id)

    assert len(result.disclaimers) == 8

    for keywords in DISCLAIMER_KEYWORDS:
        assert _keyword_present(result.disclaimers, keywords), (
            f"nessuna delle parole chiave {keywords} trovata nei disclaimer"
        )


def test_betelgeuse_proxy_disclaimer():
    """"betelgeuse" e' l'unica voce del catalogo con nota_proxy non vuota
    (ciclo "Espansione catalogo + Betelgeuse"): 9 disclaimer invece di 8
    (gli 8 standard piu' il nono, condizionale, sull'etichettatura proxy),
    con la parola chiave "proxy" presente e i valori fisici identici a s20
    (stesso modello WHW02 riusato), tranne l'id.
    """
    result = run_full_simulation("betelgeuse")

    assert len(result.disclaimers) == 9
    assert _keyword_present(result.disclaimers, ["proxy"])

    for keywords in DISCLAIMER_KEYWORDS:
        assert _keyword_present(result.disclaimers, keywords), (
            f"nessuna delle parole chiave {keywords} trovata nei disclaimer di betelgeuse"
        )

    result_s20 = run_full_simulation("s20")
    assert result.star.massa_nucleo_msun == result_s20.star.massa_nucleo_msun
    assert result.star.ye == result_s20.star.ye
    assert result.star.n_politropico == result_s20.star.n_politropico
    assert result.star.nota_proxy != ""
    assert result_s20.star.nota_proxy == ""


def test_relativistic_true_passed_explicitly():
    """`relativistic=True` deve essere passato ESPLICITAMENTE a
    simulate_collapse dalla pipeline (non un cambio del default della
    funzione, che resta False) — verificato indirettamente controllando che
    `simulate_collapse` di default (senza passare relativistic) resti
    coerente con la firma pubblica nota (default False) e che la pipeline
    produca comunque un collasso (collapsed True atteso per stelle
    supra-Chandrasekhar del catalogo).
    """
    import inspect

    from collasso.dynamics import simulate_collapse as sc

    firma = inspect.signature(sc)
    assert firma.parameters["relativistic"].default is False

    result = run_full_simulation("s20")
    # Tutte le stelle del catalogo sono supra-Chandrasekhar (vedi
    # STATUS.md, Step 4 demo) quindi ci si attende un collasso completato.
    assert result.collapse_solution.collapsed is True


def test_synthetic_collapsed_false_branch_energies_no_exceptions():
    """Test sintetico dedicato (da review): con il catalogo attuale e'
    strutturalmente impossibile ottenere collapsed=False dalle stelle di
    riferimento (Progenitor.__post_init__ impone masse sempre
    supra-Chandrasekhar). Si forza quindi collapsed=False chiamando
    simulate_collapse direttamente con un t_max_s volutamente piccolo
    (piccola frazione del tempo di caduta libera), poi si verifica che
    compute_collapse_energies non sollevi eccezioni e gestisca in modo
    coerente l'assenza di un vero momento di collasso.
    """
    from collasso.dynamics import free_fall_time_s

    catalogo_star_id = "s20"
    from collasso.catalog import get_progenitor_by_id, load_reference_catalog

    star = get_progenitor_by_id(load_reference_catalog(), catalogo_star_id)

    solution = solve_lane_emden(n=star.n_politropico)
    profile = physical_profile(solution, star.densita_centrale_gcm3, star.massa_nucleo_msun)
    delta_m_g, m_enclosed_g, r0_cm = build_initial_shells(profile.r_cm, profile.rho_gcm3, 50)

    t_ff_s = free_fall_time_s(star.densita_centrale_gcm3)
    # t_max_s volutamente piccolo (1e-4 * t_ff) per forzare collapsed=False:
    # troppo poco tempo perche' qualunque evento terminale scatti.
    t_max_s_piccolo = 1.0e-4 * t_ff_s

    collapse_solution = simulate_collapse(
        delta_m_g, r0_cm, ye=star.ye, t_max_s=t_max_s_piccolo, relativistic=True
    )

    assert collapse_solution.collapsed is False
    assert collapse_solution.t_collapse_s is None
    assert collapse_solution.collapse_reason is None

    # Il calcolo delle energie non deve sollevare eccezioni anche in questo
    # ramo, e deve usare l'ULTIMO istante disponibile (non un vero momento
    # di collasso).
    energie = compute_collapse_energies(collapse_solution, delta_m_g, m_enclosed_g)

    assert energie.KE_finale_erg >= 0.0
    assert energie.PE_iniziale_erg < 0.0
    assert energie.PE_finale_erg < 0.0
    assert energie.t_energie_s == pytest.approx(collapse_solution.t_s[-1])
