# Progetto: Simulazione numerica del collasso gravitazionale stellare

## Obiettivo
Simulazione a simmetria sferica del collasso di una stella, inizializzata da
parametri di stelle reali (catalogo progenitori), con output sia numerico
(velocita, tempi scala, energie, classificazione del remnant) sia visivo
(animazione dell'evoluzione).

## Dati iniziali
Fonte: cataloghi pubblici di modelli progenitori (Sukhbold, Woosley & Heger).
Parametri estratti per ogni stella: massa del nucleo, raggio iniziale (o
densita centrale), frazione di elettroni Ye, indice politropico n.

## Vincoli fisici (non negoziabili — ogni modifica al modello va verificata contro questa tabella)

| Vincolo | Riferimento |
|---|---|
| Profilo di equilibrio iniziale | Equazione di Lane-Emden (politropica) |
| Pressione di degenerazione | EOS di Chandrasekhar (non-rel. -> ultra-rel.) |
| Soglia di instabilita | Massa di Chandrasekhar (~1.4 Msun) |
| Dinamica del collasso | Shell Lagrangiane, simmetria sferica |
| Gravita | Newtoniana + correzioni relativistiche sul nucleo |
| Classificazione remnant | Limite TOV, EOS nucleare realistica via piecewise polytrope (Read, Lackey, Owen & Friedman 2009, PRD 79, 124032, arXiv:0812.2163 — fit a SLy/APR4, accurata entro ~1% sulle relazioni massa-raggio rispetto ai modelli tabulati). Il gas di neutroni liberi (Oppenheimer & Volkoff 1939) resta implementato come caso limite storico/didattico, non più il default per la classificazione. |
| Neutrini | Termine di raffreddamento semplificato (NON trasporto completo) — limite dichiarato |
| Rotazione / campo magnetico | Trascurati — limite dichiarato |
| Validazione | Confronto qualitativo con GR1D (github.com/evanoconnor/GR1D) |

## Sub-agenti del progetto (in agents/)
- planner: scompone il progetto in step, aggiorna STATUS.md
- reviewer: secondo controllo critico su piani e report
- coder: implementa il codice dello step corrente
- critic-fisico: verifica il codice contro la tabella dei vincoli sopra
- optimizer: pulizia codice, stile, ridondanze (modello economico)
- reporter: scrive il resoconto giornaliero sintetico (modello economico)

## Regole di lavoro
- Prima di implementare un nuovo step, usa la plan mode per proporre l'approccio.
- Filtra sempre l'output numerico verboso (log di simulazione) prima che
  arrivi nel contesto principale: passa attraverso un subagent o un comando
  che estrae solo le metriche rilevanti.
- Ogni ciclo giornaliero: leggi STATUS.md, esegui il prossimo step, aggiorna
  STATUS.md, genera il report con il subagent reporter, fermati e attendi
  conferma dell'utente.
- Nessun push pubblico su GitHub senza conferma esplicita dell'utente.
