# Budget-System

Diese Datei beschreibt die aktuelle Budget-Implementierung im Backend des Fast OpenAI API Proxy.

Stand: 2026-05-11

## Kurzüberblick

Das Budget-System begrenzt die Nutzung von `/v1/*`-Requests anhand von Credits. Budgets werden in SQLite gespeichert, zur Laufzeit in Memory gespiegelt und pro Anfrage erst reserviert und nach der Upstream-Antwort mit den tatsächlichen Kosten verrechnet.

Die wichtigsten Bausteine sind:

- `backend/app/middleware/access_control.py`: entscheidet vor `/v1/*`-Requests, welche Identität belastet wird und ob genügend Budget vorhanden ist.
- `backend/app/budget_service.py`: hält In-Memory-Verbrauch, reserviert Credits und schreibt finalen Verbrauch asynchron weg.
- `backend/app/utils.py`: berechnet nach der Provider-Antwort Usage und Kosten.
- `backend/app/usage_worker.py`: persistiert Request-Logs und Budget-Verbrauch im Hintergrund.
- `backend/app/access_store.py`: enthält die SQLite-Tabellen und CRUD-Operationen für Budgets, Verbrauch und Request-Logs.
- `backend/app/models_handler.py` und `backend/app/config_store.py`: liefern pro Modell `type`, `price_per_unit` und `min_credits_per_request`.

## Aktivierung

Budget-Prüfung hängt an der Access-Control-Middleware. Diese wird nur aktiviert, wenn:

```env
FOAP_ENABLE_ACCESS_CONTROL=true
```

gesetzt ist. Ohne diese Middleware werden Budgets nicht vor Requests geprüft. Die spätere Kostenauflösung in `utils.py` setzt ebenfalls voraus, dass die Middleware vorher `request.state.payer_entity` gesetzt hat.

Optionale Debug-Ausgabe für Budget-/Quota-Entscheidungen:

```env
FOAP_ENABLE_QUOTA_DECISION_TRACE=true
```

oder request-spezifisch:

```http
X-FOAP-Debug-Quota: 1
```

Dann enthält die Antwort bei Middleware-Entscheidungen den Header:

```http
X-FOAP-Quota-Decision: source=budget;allowed=True;api_path=/v1/...;model=...;owner=...
```

## Datenmodell

Die Budget-Daten liegen in der Access-DB. Der Pfad kommt aus:

```env
FOAP_ACCESS_DB_PATH
```

Default ist:

```text
data/access.db
```

### Tabelle `budgets`

Definiert Limits.

Spalten:

- `id`: UUID des Budgets.
- `entity_type`: aktuell per Schema `user` oder `group`.
- `entity_id`: ID des Users oder der Gruppe.
- `scope`: optionaler Modelltyp-Scope. `NULL`/leer bedeutet global für alle Modelltypen.
- `window`: `daily` oder `monthly`.
- `budget_amount`: erlaubte Credits im Fenster.
- `created_at`: Unix-Zeitstempel.

Eindeutigkeit:

```text
UNIQUE(entity_type, entity_id, scope, window)
```

Wichtig: SQLite behandelt `NULL` in Unique-Constraints speziell. Da beim Schreiben von Usage leere Scopes als `""` normalisiert werden, globale Budgets können je nach Erzeugung mit `scope = NULL` oder `scope = ""` auftreten.

### Tabelle `budget_usage`

Aggregierter Verbrauch pro Entität, Scope und Zeitfenster.

Spalten:

- `entity_type`
- `entity_id`
- `scope`
- `window`
- `window_bucket`: bei daily `YYYY-MM-DD`, bei monthly `YYYY-MM`.
- `cost`: aufsummierte tatsächliche Kosten.

Primary Key:

```text
PRIMARY KEY(entity_type, entity_id, scope, window, window_bucket)
```

### Tabelle `request_logs`

Append-only Request-Protokoll für spätere Auswertung.

Spalten:

- `api_key_id`
- `timestamp`
- `requested_model`
- `target_model_name`
- `provider`
- `scope`
- `usage`
- `usage_unit`
- `price`
- `price_per_unit`
- `cost`

Aktuell gibt es dafür ein Pydantic-Schema (`RequestLogRead`), aber keinen öffentlichen Read-Endpunkt.

## Modell-Konfiguration für Kosten

Kosten werden aus der Provider-/Modellkonfiguration gelesen. Relevant sind pro Modell:

- `type`: Modelltyp, Default `llm`.
- `price_per_unit`: Credit-Kosten pro Usage-Einheit.
- `min_credits_per_request`: Mindestbetrag, der vor dem Request reserviert wird.

Diese Felder liegen in `provider_models` (`config_store.py`) und werden in `models_handler.py` in `model_data` übernommen.

Beispielhafte Modellwerte:

```json
{
  "type": "llm",
  "price_per_unit": 0.00001,
  "min_credits_per_request": 0.01
}
```

`min_credits_per_request` steuert die harte Vorabprüfung. Wenn dieser Wert `0` ist, wird kein Budget reserviert und damit auch kein `payer_entity` gesetzt; der aktuelle Budget-Accounting-Pfad greift dann praktisch nicht.

## Identitäts- und Payer-Auflösung

Die Middleware baut eine priorisierte Liste möglicher zu belastender Entitäten (`entities`):

1. Wenn ein gültiger OIDC-Owner erkannt wird:
   - `("user", oidc_owner_id)`
   - danach alle OIDC-Gruppen als `("group", group_id)`
2. Sonst, wenn ein FOAP API-Key mit `owner_id` erkannt wird:
   - `("user", owner_id)`
3. Sonst, wenn ein FOAP API-Key ohne Owner erkannt wird:
   - `("user", "key:<api_key_id>")`

Diese Reihenfolge ist relevant: `BudgetService.reserve_budget(...)` prüft die Entitäten nacheinander und nimmt die erste Entität, die alle passenden Budgets erfüllen kann. Diese Entität wird als `request.state.payer_entity` gespeichert.

Wenn ein geschützter Endpoint ohne Identität angefragt wird, antwortet die Middleware mit `401 Unauthorized`.

## Request-Flow

### 1. Middleware liest Request

Für Nicht-GET-Requests versucht `AccessControlMiddleware`, den JSON-Body zu lesen und daraus `model` zu bestimmen. Danach wird der Body für downstream Handler wiederhergestellt.

Nur Pfade unter `/v1/` werden budgetrelevant geprüft.

### 2. Modell wird aufgelöst

Wenn ein `model` vorhanden ist, ruft die Middleware:

```text
models_handler.get_model_data(model, path)
```

auf. Dadurch werden Alias-/Modellkonfiguration, unterstützter Endpoint, Modelltyp, Preis und Mindestcredits bestimmt.

### 3. Mindestcredits werden reserviert

Wenn `min_credits_per_request > 0` und Identitäten vorhanden sind, ruft die Middleware:

```text
budget_service.reserve_budget(entities, model_type, min_credits)
```

auf.

Die Budget-Auswahl ist:

- Es werden nur Budgets der jeweiligen Entität geladen.
- Ein Budget ist anwendbar, wenn:
  - `scope` leer/global ist, oder
  - `scope == model_type`.
- Für jedes anwendbare Budget wird das passende Bucket genutzt:
  - `daily` -> `YYYY-MM-DD`
  - `monthly` -> `YYYY-MM`
- Wenn `used + min_credits > budget_amount`, kann diese Entität den Request nicht bezahlen.
- Wenn alle anwendbaren Budgets passen, werden `min_credits` sofort im In-Memory-Verbrauch addiert.

Bei Erfolg setzt die Middleware:

- `request.state.payer_entity`
- `request.state.reserved_credits`
- `request.state.model_data`

Bei Misserfolg antwortet sie mit:

```http
429 Budget exhausted
```

### 4. Request geht zum Upstream Provider

Der eigentliche Proxy-Fluss bleibt in `utils.py` erhalten:

- Request wird an Provider weitergeleitet.
- Streaming und Non-Streaming werden separat behandelt.
- Responses API kann intern auf Chat Completions übersetzt werden.
- Fallback-Modell-Routing und Provider-Rate-Limit-Sync bleiben Teil des Flusses.

### 5. Tatsächliche Kosten werden berechnet

Nach Antwort oder Fehler ruft `utils.py` `_resolve_budget(...)` auf. Diese Funktion ist idempotent pro Request über `request.state.budget_resolved`.

Kostenformel:

```text
cost = final_usage * price_per_unit
```

Usage-Ermittlung nach Modelltyp:

| Modelltyp | Usage | Unit |
| --- | --- | --- |
| `llm` | `usage.total_tokens` aus JSON/SSE, sonst `0` | `tokens` |
| `embedding` | `usage.total_tokens` aus JSON/SSE, sonst `0` | `tokens` |
| `image-gen` | Request-Feld `n`, sonst `1` | `images` |
| `stt` | explizite Usage, sonst Request-Laufzeit | `seconds` |
| `tts` | explizite Usage, sonst Request-Laufzeit | `seconds` |
| `realtime` | explizite Usage, sonst Request-Laufzeit | `seconds` |
| sonstige | explizite Usage, sonst `1` | `units` |

Bei Upstream-Fehlern (`is_error=True`) bleibt `final_usage = 0.0`; dadurch werden die reservierten Credits später wieder aus dem In-Memory-Verbrauch herausgerechnet.

### 6. Reservation wird auf Ist-Kosten angepasst

`BudgetService.resolve_budget(...)` berechnet:

```text
adjustment = actual_cost - min_credits
```

Dann wird der In-Memory-Verbrauch für alle passenden Budgets angepasst.

Beispiele:

- Reserviert `0.01`, tatsächliche Kosten `0.03` -> In-Memory +`0.02`.
- Reserviert `0.01`, tatsächliche Kosten `0.00` -> In-Memory -`0.01`.
- Reserviert `0.01`, tatsächliche Kosten `0.005` -> In-Memory -`0.005`.

Persistiert wird nicht das Adjustment, sondern der tatsächliche Cost-Wert.

### 7. Asynchrone Persistierung

`usage_worker.py` läuft ab FastAPI-Lifespan-Start als Hintergrundtask.

Er verarbeitet zwei Queue-Typen:

- `log_request`: schreibt einen Datensatz in `request_logs`.
- `add_budget_usage`: addiert Kosten in `budget_usage`.

Beim Shutdown wird `None` in die Queue gelegt, der Worker beendet sich danach.

## In-Memory-Verbrauch

`BudgetService` hält Verbrauch in:

```text
memory_usage[entity_type][entity_id][window][window_bucket][model_type_or_scope] = cost
```

Beim ersten Zugriff auf eine Entität wird bestehende DB-Usage aus `budget_usage` geladen (`_init_entity`). Danach arbeitet die Vorabprüfung gegen Memory, damit parallele Requests sofort berücksichtigt werden.

Für jede Entität gibt es einen eigenen `asyncio.Lock`, um Race Conditions bei Reservation und Adjustment innerhalb eines laufenden Prozesses zu reduzieren.

Wichtig: Der In-Memory-Cache ist pro Prozess. Bei mehreren Worker-Prozessen oder mehreren Instanzen ist die Vorabreservierung nicht global synchronisiert; die DB wird nur asynchron nachgezogen.

## Admin-API

Die Budget-Verwaltung hängt an der Admin-API und ist nur verfügbar, wenn:

```env
FOAP_ENABLE_ADMIN_API=true
```

und Admin-Auth korrekt konfiguriert ist.

Endpoints:

```http
GET    /api/admin/budgets
GET    /api/admin/budgets?entity_type=user&entity_id=<id>
GET    /api/admin/budgets/{budget_id}
POST   /api/admin/budgets
PUT    /api/admin/budgets/{budget_id}
DELETE /api/admin/budgets/{budget_id}
```

Create-Payload:

```json
{
  "entity_type": "user",
  "entity_id": "alice",
  "scope": "llm",
  "window": "monthly",
  "budget_amount": 100.0
}
```

`scope` kann `null` sein, um ein globales Budget für alle Modelltypen zu definieren.

Schema-Regeln:

- `entity_type`: `user` oder `group`
- `window`: `daily` oder `monthly`
- `budget_amount`: `>= 0`

## Self-Service-API

Die Self-Service-Budgetanzeige ist verfügbar, wenn:

```env
FOAP_ENABLE_SELF_SERVICE_API=true
```

Endpoints:

```http
GET /api/budgets
GET /api/budgets/usage
```

Diese Endpoints zeigen nur Budgets und Usage für den aktuell ermittelten User:

```text
store.list_budgets(entity_type="user", entity_id=owner_id)
store.get_all_budget_usage(entity_type="user", entity_id=owner_id)
```

Gruppenbudgets werden in der aktuellen Self-Service-Ansicht nicht aggregiert angezeigt, obwohl sie bei der Middleware-Entscheidung als Payer infrage kommen.

## Frontend-Integration

Admin:

- `frontend/src/views/BudgetsView.vue`
- ruft `/api/admin/budgets` auf.
- ermöglicht Erstellen, Ändern und Löschen von Budgets.

Self-Service:

- `frontend/src/views/AccountView.vue`
- ruft `/api/budgets` und `/api/budgets/usage` auf.
- zeigt Budgetkarten und Fortschrittsbalken an.

Aktueller Hinweis: `AccountView.vue` sucht Usage mit `u.budget_id === budget.id`. Backend-Usage-Datensätze enthalten aber kein `budget_id`, sondern `entity_type`, `entity_id`, `scope`, `window`, `window_bucket` und `cost`. Dadurch kann die Self-Service-Anzeige aktuell Verbrauch als `0` darstellen, obwohl `budget_usage` Einträge enthält.

## Scope- und Modelltyp-Konventionen

Backend-relevante Scopes entsprechen `model_data["type"]`.

Aktuell im Code verwendete Typen:

- `llm`
- `embedding`
- `image-gen`
- `stt`
- `tts`
- `realtime`
- beliebige weitere Strings als Fallback-`units`

Wichtig: Das Admin-Frontend bietet derzeit u. a. `image`, `audio_transcription` und `audio_speech` an. Diese Werte matchen nicht automatisch die Backend-Typen `image-gen`, `stt` oder `tts`. Für wirksame scoped Budgets muss der Scope exakt dem Modelltyp entsprechen.

## Fehler- und Edge-Case-Verhalten

- Wenn die Middleware bei Modellauflösung, Budgetladung oder Reservation eine Exception bekommt, wird sie aktuell still geschluckt (`except Exception: pass`). Der Request läuft dann ohne Budgetblockade weiter.
- Wenn kein passendes Budget für eine Entität existiert, wird diese Entität übersprungen. Existiert für keine Entität ein passendes Budget, schlägt `reserve_budget` fehl, sofern `min_credits_per_request > 0` ist.
- Globale Budgets (`scope` leer) und typ-spezifische Budgets können gleichzeitig gelten. Dann müssen alle anwendbaren Budgets genügend Restbudget haben.
- Bei Streaming-LLM-Responses wird `stream_options.include_usage = true` injiziert, damit `usage.total_tokens` aus SSE gelesen werden kann.
- Bei Subresource- und Upload-Streaming wird derzeit keine tokenbasierte Usage extrahiert; je nach Typ fällt die Berechnung auf Laufzeit oder Default-Units zurück.
- Die persistente Budget-Usage wird asynchron geschrieben. Direkt nach einem Request kann die DB kurz hinter dem In-Memory-Stand zurückliegen.
- Die Reservation schützt nur innerhalb eines Python-Prozesses zuverlässig gegen Parallelität. Mehrere Serverinstanzen teilen diese Reservation nicht synchron.

## Beispielablauf

Angenommen:

- User `alice`
- Modell `gpt-4o-mini`
- `type = "llm"`
- `price_per_unit = 0.00001`
- `min_credits_per_request = 0.01`
- Monatsbudget für `alice`, Scope `llm`, Betrag `10.0`

Ablauf:

1. Request `POST /v1/chat/completions` mit `model = "gpt-4o-mini"` kommt rein.
2. Middleware erkennt `alice` als User.
3. Modell wird aufgelöst, Mindestcredits sind `0.01`.
4. Budgetservice prüft Monatsbucket `2026-05`.
5. Wenn bisher z. B. `3.50` Credits verbraucht wurden, ist `3.50 + 0.01 <= 10.0`; Request wird erlaubt.
6. `0.01` Credits werden in Memory reserviert.
7. Provider antwortet mit `usage.total_tokens = 2400`.
8. Kosten: `2400 * 0.00001 = 0.024`.
9. In-Memory-Verbrauch wird um `0.024 - 0.01 = 0.014` erhöht.
10. Worker schreibt `0.024` in `budget_usage` und einen Request-Log in `request_logs`.

## Relevante Dateien

- `backend/app/main.py`: startet `usage_worker_task`, aktiviert Middleware abhängig von `FOAP_ENABLE_ACCESS_CONTROL`.
- `backend/app/middleware/access_control.py`: Identitätsauflösung, Budget-Reservation, 429 bei erschöpftem Budget.
- `backend/app/budget_service.py`: In-Memory-Usage, Locks, Reservation, finale Anpassung.
- `backend/app/utils.py`: Usage-/Kostenberechnung nach Upstream-Response, Request-Logging.
- `backend/app/usage_worker.py`: asynchroner DB-Writer.
- `backend/app/access_store.py`: SQLite-Schema und Budget-/Usage-/Log-Operationen.
- `backend/app/models_handler.py`: Modellauflösung und Weitergabe von `type`, `price_per_unit`, `min_credits_per_request`.
- `backend/app/schemas/access.py`: Pydantic-Schemas für Budget-CRUD und Budget-Usage.
- `backend/app/routers/admin.py`: Admin-Budget-Endpoints.
- `backend/app/routers/self_service.py`: Self-Service-Budget- und Usage-Endpoints.
- `frontend/src/views/BudgetsView.vue`: Admin-UI.
- `frontend/src/views/AccountView.vue`: Self-Service-Anzeige.
