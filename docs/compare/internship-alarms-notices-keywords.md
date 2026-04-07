# 인턴십 프로젝트: 알람 · 공지사항 · 키워드 구독 — 코드 흐름 정리

이 문서는 **코드 초보자**도 따라갈 수 있도록, 관련 파일과 **함수 이름**, **데이터가 어디서 어디로 가는지**를 순서대로 설명한다.  
실제 파일 경로는 저장소 `internship` 기준이며, 백엔드는 `backend/app/`, 프론트는 `frontend/src/` 아래에 있다.

> 참고: 요청하신 `core/alarmworker.py`는 저장소에는 **`core/alarm_worker.py`**(언더스코어)로 존재한다.  
> `schemas/notice.py`는 **`schemas/req/notice.py`**에 있다.

---

## 1. 한눈에 보는 전체 흐름

1. **어딘가에서 이벤트 발생** (공지 작성, 파일 업로드, 댓글, 권한 변경 등)
2. **`AlarmService.create_alarm`** 이 DB 테이블 `alarms`에 행을 추가하고, 동시에 **Redis 채널 `alarm_channel`** 로 JSON 메시지를 발행한다.
3. 서버가 뜰 때 **`redis_alarm`** 백그라운드 태스크가 Redis를 구독하다가 메시지를 받으면 **`alarm_manager.send_personal_message`** 로 해당 사용자의 WebSocket들에 JSON을 push한다.
4. 브라우저는 **`DefaultLayout`** 에서 **`useAlarmSocket`** 으로 `ws://.../api/alarms/ws?token=...` 에 연결해 두고, 메시지가 오면 Redux **`addAlarm`** 으로 목록 맨 앞에 넣는다.
5. 사용자가 사이드바를 열면 **`fetchAlarmThunk`** 로 REST **`GET /api/alarms/`** 를 호출해 DB에 쌓인 알람을 다시 가져온다.
6. **공지(SYSTEM_ALERT)** 는 알람 클릭 시 **`GET /api/notice/{id}`** 로 본문을 불러와 **`NoticeModal`** 로 보여준다.
7. **키워드 구독**은 마이페이지에서 **`/api/mypage/keywords`** 로 저장되고, **다른 사람이 해당 대분류(`bc_id`)로 파일을 올리거나 제목을 바꿀 때** `files.py` 가 구독자에게 `KEYWORD_MATCH` / `DATA_UPDATE` 알람을 만든다.

---

## 2. 데이터베이스 모델 (백엔드)

### 2.1 `model/alarms.py`

- **`AlarmEnum`**  
  알람 종류 문자열. 예: `SYSTEM_ALERT`, `KEYWORD_MATCH`, `COMMENT_ADDED`, `AUTH_CHANGED`, `CHAT_MESSAGE`, `RANK_CHANGED` 등.

- **`Alarm` 테이블 `alarms`**
  - `user_id`: 알람 받을 사용자
  - `title`: 알람 제목 (예: "공지사항", "관심 키워드 알림")
  - `content`: **중요** — 보통 `"본문메시지|참조ID"` 형태로 저장된다. `참조ID`가 없으면 메시지만 저장.
  - `is_read`: 읽음 여부
  - `created_at`: 생성 시각
  - `alarm_type`: `AlarmEnum`

> 코드상 SQLModel에는 `ref_id` 컬럼이 없다. 참조 ID는 **`AlarmService`가 `content`에 `|` 로 붙여 넣는 방식**이다.  
> `routers/alarms.py`의 `hasattr(alarm, "ref_id")` 는 나중에 컬럼을 추가했을 때를 대비한 코드로 보이며, **현재 모델만 보면 항상 거짓**일 수 있다.

### 2.2 `model/notices.py`

- **`Notice`**: 공지 글. `user_id`, `title`, `content`(LONGTEXT), `access_level`, `created_at`.
- **`AccessEnum`**: `"1"`, `"2"`, `"3"` — 열람 최소 권한 수준.

### 2.3 `model/keywords.py`

- **`UserKeyword`**: 사용자가 구독한 **대분류** 하나당 한 행.
  - `user_id`, `bc_id` (foreign key → `big_ctgrs.id`)
- “키워드”라는 이름이지만, 실제로는 **대분류(빅카테고리) 구독**에 가깝다.

---

## 3. 알람 생성·전달 핵심 (백엔드)

### 3.1 `service/alarm_service.py` — `AlarmService`

| 메서드 | 하는 일 |
|--------|---------|
| `__init__(db)` | DB 세션 보관, Redis 채널 이름 `"alarm_channel"` 고정 |
| **`create_alarm(user_id, title, message, alarm_type, ref_id=None)`** | ① `ref_id`가 있으면 `content = f"{message}\|{ref_id}"`, 없으면 `content = message`. ② `Alarm` 행 insert + commit. ③ Redis `publish`에 넣을 dict: `id`, `user_id`, `title`, `message`(원문), `target_id`(=ref_id), `created_at`, `type`, `is_read=False`. |

**초보자 포인트**: DB에는 `message|id`가 합쳐진 문자열이 들어가고, Redis/WebSocket으로 가는 JSON에는 `message`와 `target_id`가 **분리**되어 있다.

### 3.2 `core/alarm_worker.py` — `redis_alarm()`

- `redis_client.pubsub()` 로 **`alarm_channel` 구독**.
- 무한 루프에서 `get_message`로 메시지 수신 → JSON 파싱 → `user_id` → **`await alarm_manager.send_personal_message(target_user_id, data)`**.
- 서버 종료 시 unsubscribe/close.

**역할**: “DB에 알람이 생김”과 “지금 로그인해서 WS 연결한 브라우저에 즉시 푸시”를 잇는 **다리**.

### 3.3 `websocket/alarm_manager.py` — `AlarmManager`

| 메서드 | 하는 일 |
|--------|---------|
| `connect(user_id, websocket)` | `activate_connections[user_id]` 리스트에 소켓 추가 (한 사용자 여러 탭 가능) |
| `disconnect(user_id, websocket)` | 리스트에서 제거, 비면 키 삭제 |
| `send_personal_message(user_id, data)` | 해당 사용자의 모든 WS에 `send_json(data)` 시도, 실패는 무시 |

전역 싱글톤 **`alarm_manager`** 가 `alarms` 라우터와 `redis_alarm`에서 공유된다.

### 3.4 `main.py`에서의 기동

`@app.on_event("startup")` 안에서:

- `asyncio.create_task(redis_alarm())` — 위 Redis 구독 루프 실행

---

## 4. REST API: 알람 — `routers/alarms.py`

프리픽스: **`/api/alarms`**

| HTTP | 경로 | 함수명 | 설명 |
|------|------|--------|------|
| GET | `/` | `get_alarms` | 로그인 사용자 알람 목록. **공지(`SYSTEM_ALERT`)** 는 최대 30개(날짜 제한 없음). **그 외** 는 최근 **3일 이내**만, 최대 30개. 합쳐 시간순 정렬. `content`를 `\|`로 나눠 `message`, `target_id` 복원. 공지인데 대응 `Notice`가 없으면 알람 행 **삭제** 후 목록에서 제외. |
| PATCH | `/{alarm_id}/read` | `alarm_as_read` | 단건 읽음 처리 |
| PATCH | `/read-all/{user_id}` | `read_all_alarms` | 쿼리 `alarm_type`: `notice` → 공지만, `general` → 공지 제외, 그 외 `AlarmEnum` 이름이면 해당 타입만 전부 읽음 |
| DELETE | `/delete-all/{user_id}` | `delete_all_alarms` | **공지가 아닌** 알람 전부 삭제 |
| DELETE | `/delete/{alarm_id}` | `delete_alarm` | 단건 삭제 |
| WebSocket | `/ws` | `websocket_alarm_endpoint` | 쿼리 `token`으로 `get_current_user_ws` 인증 후 `alarm_manager.connect` |

---

## 5. REST API: 공지 — `routers/notice.py`

프리픽스: **`/api/notice`**

| HTTP | 경로 | 함수명 | 설명 |
|------|------|--------|------|
| GET | `/list` | `get_notice_list` | 페이지·검색·날짜 필터. 비관리자는 `access_level`과 본인 글 규칙으로 필터. |
| GET | `/{notice_id}` | `get_notice_detail` | 단건 + 권한 검사 |
| POST | `/write` | `create_notice` | 공지 저장 후, `access_level`에 따라 **열람 가능한 역할**을 가진 **활성 사용자 전원**에게 루프로 `AlarmService.create_alarm(..., SYSTEM_ALERT, ref_id=new_notice.id)` |
| DELETE | `/{notice_id}` | `delete_notice` | 작성자/관리자 규칙으로 삭제 |

---

## 6. 키워드(대분류) 구독 — `routers/mypage.py`

| HTTP | 경로 | 함수명 | 설명 |
|------|------|--------|------|
| GET | `/keywords` | `get_keywords` | `BigCtgr` 중 이름이 `"미정"`이 아닌 목록 + 현재 사용자의 `user_keywords`의 `bc_id` 리스트 |
| POST | `/keywords` | `update_keywords` | 기존 `UserKeyword` 전부 삭제 후, 요청의 `keyword_ids`(실제로는 **bc_id 리스트**)로 다시 insert |

프론트 `Mypage.jsx`는 이 API를 **`/mypage/keywords`** 로 호출한다(axios base URL에 `/api`가 붙는 구조일 수 있음 — 인터셉터 설정 확인).

---

## 7. 알람을 “만드는” 다른 라우터들

### 7.1 `routers/files.py`

- **파일 저장 성공 후** (요약 완료 알람 코드는 **주석 처리**됨 — `SUMMARY_COMPLETE`).
- **`KEYWORD_MATCH`**: 새 파일의 `bc_id`가 있으면, `UserKeyword.bc_id`가 같고 본인이 아닌 사용자들에게 알람. `ref_id=new_file.id` → 상세 `/details/{id}` 로 이동에 사용.
- **`rename_file`**: 제목 변경 시 같은 방식으로 구독자에게 `DATA_UPDATE` 알람.

### 7.2 `routers/bbs.py` — `create_new_comment`

- 게시글 작성자에게 (본인 댓글·멘션 제외 시) `COMMENT_ADDED`, `ref_id=file_id`.
- `mentioned_ids` 각각에게 `COMMENT_MENTION`, `ref_id=file_id`.

### 7.3 `routers/users.py` — 테트리스 점수

- 최고 점수 갱신 시 **이전 1위 `old_user_id`** 에게 `RANK_CHANGED` 알람 (`ref_id` 없음).

### 7.4 `routers/cms.py` — 권한 변경

- 대상 사용자에게 `AUTH_CHANGED`, `ref_id=user.id` (메시지에 이전/이후 등급 이름).

### 7.5 `routers/chat.py`

- 채팅 메시지에 `mentions`가 있으면 멘션된 사용자들에게 `CHAT_MESSAGE` 알람 (**`ref_id` 없음** — 프론트 `alarmConfig`의 `CHAT_MESSAGE.path`는 `?chatRoomId=` 를 쓰는데 ID가 안 맞을 수 있음).

---

## 8. 스키마 `schemas/req/notice.py`

- **`Pagination`**: 총건수·페이지 수·그룹 네비게이션 필드를 `model_validator`로 계산.
- **`NoticeCreate`**: 작성 시 `title`, `content`, `access_level`.
- **`NoticeRead`**, **`NoticeListResponse`**: API 응답 형태 정의.

---

## 9. 프론트엔드 — HTTP 래퍼

### 9.1 `components/http/alarmHttp.js`

- `getAlarms()` → `GET /alarms/` (axios base에 `/api` 포함 시 실제 `/api/alarms/`)
- `markAsRead`, `markAllAsRead`, `deleteAlarm`, `deleteAllAlarm`

**참고**: `markAllAsRead`에 인자로 **`${userId}?alarm_type=notice`** 처럼 문자열을 넘기면, URL이 `.../read-all/5?alarm_type=notice` 가 되어 FastAPI가 `user_id=5`, 쿼리 `alarm_type=notice`로 받는다 (`SideBar.jsx` 패턴).

### 9.2 `components/http/noticeHttp.js`

- `selectNoticeList` → `GET /notice/list`
- `insertNotice` → `POST /notice/write`

---

## 10. 프론트엔드 — Redux

### 10.1 `stores/thunks/alarmThunk.js`

- **`fetchAlarmThunk`**: `alarmHttp.getAlarms()` — 인자 `userId`는 HTTP에서 안 쓰이지만 thunk 시그니처에 남아 있음.
- **`markAsReadThunk`**, **`markAllAsReadThunk`**, **`deleteAlarmThunk`**, **`markAllDeleteThunk`**
- **`alarmSlice`**: `list`, `unreadCount`, `addAlarm`(WS용), `extraReducers`로 fulfilled 시 목록/읽음/삭제 반영.

### 10.2 `stores/thunks/noticeThunk.js`

- **`readNoticeList`**: `noticeAction`으로 로딩·목록·페이지네이션·`bigCtgrs` 설정.
- **`writeNotice`**: `insertNotice` (다른 화면용; `NoticeList`는 직접 `api.post`도 사용).

---

## 11. 프론트엔드 — WebSocket · 레이아웃

### 11.1 `components/ui/useAlarmSocket.js`

- `userId`와 JWT가 있을 때 `VITE_WS_URL`(기본 `ws://localhost:8000`) + `/api/alarms/ws?token=...` 연결.
- `onmessage` → JSON 파싱 → **`dispatch(addAlarm(newAlarm))`**.

### 11.2 `components/ui/DefaultLayout.jsx`

- **`useAlarmSocket(user?.id)`** 호출 — 로그인한 모든 주요 화면에서 WS 유지.
- `unreadCount` 등으로 배지 표시 가능.

---

## 12. 프론트엔드 — UI 컴포넌트

### 12.1 `components/ui/alarmConfig.js`

- 알람 **`type`** 문자열 → **라벨 색**, **클릭 시 이동 경로 함수** `path(id)`.
- `SYSTEM_ALERT`, `AUTH_CHANGED` 는 `path: null` (모달 등 별도 처리).

### 12.2 `components/ui/AlarmList.jsx`

- props: `alarms`, `onItemClick`, `isLoading`, `emptyMessage`.
- 내부에서 다시 **공지 / 비공지** 필터·**3일 제한**을 걸어 표시(백엔드와 유사 규칙).
- 클릭 시: `onItemClick` → **`markAsReadThunk`** → `getHistories()` → `ALARM_CONFIG` 기반 `navigate`.
- 읽은 일반 알람만 **개별 삭제** 버튼.

### 12.3 `components/ui/ToastBanner.jsx`

- Redux `alarms` 목록의 **최신 건** id가 올라오면 토스트 추가 (채팅 탭 열림 + `CHAT_MESSAGE` 는 토스트 생략·자동 읽음 처리 등).
- 클릭 시 읽음 + 경로 이동.

### 12.4 `components/ui/SideBar.jsx` (알람·공지 UI의 중심)

- `fetchAlarmThunk()` 로 목록 로드.
- 뷰 모드: 메뉴 / **공지 탭** (`noticeList`) / **전체 알림 탭** (`generalAlarmList`).
- **`handleAlarmClick`**:
  - `SYSTEM_ALERT`: `GET /notice/{targetId}` 로 본문 받아 **`NoticeModal`** 또는 에러 메시지 모달.
  - `AUTH_CHANGED`: 읽음 + 모달.
  - `CHAT_MESSAGE`: 우측 패널 채팅 탭 오픈.
  - `RANK_CHANGED`: `/mini2` 이동.
- **전체 읽기**: 공지 탭은 `markAllAsReadThunk(\`${user.id}?alarm_type=notice\`)`, 알림 탭은 `...general`.
- **`ToastBanner`** 에 `alarms` 전달.

### 12.5 `components/ui/NoticeModal.jsx`

- 범용 모달: `title`, `text`, `onConfirm`, `onCancle`(오타), 포털 렌더, ESC로 닫기.
- 공지 **본문 표시**는 SideBar에서 API로 받은 `title`/`content`를 넘겨줄 때 사용.

### 12.6 `components/forum/notice/NoticeList.jsx`

- 공지 **목록 페이지**: 검색·페이지네이션·작성 모달·삭제.
- 작성 성공 시 `fetchAlarmThunk()` + `getHistories()` — 새 공지 알람 반영.
- `NoticeModal` / `AlertModal` / `ConfirmModal` 사용.

### 12.7 `components/forum/mypage/Mypage.jsx` (키워드 부분)

- **`keyword()`**: `GET .../mypage/keywords` → `keywordList`, `selectedKeywords` 세팅 후 모달 오픈.
- **`handleKeywordToggle`**: 선택 토글 (bc_id).
- **`onSaveKeywords`**: `POST .../mypage/keywords` + `{ keyword_ids: selectedKeywords }`.
- 마운트 시 `useEffect`로 키워드 목록 한 번 더 로드해 `savedKeywords` 표시.

---

## 13. `components/forum/history` 폴더

- **`Histories.jsx`**, **`Details.jsx`** 등은 **업로드 문서 목록·상세** UI다.
- 이 폴더 안에는 **알람/공지/키워드 전용 API 호출 코드가 없다.**
- 다만 알람의 `KEYWORD_MATCH` / `DATA_UPDATE` / 댓글 알람이 **`/details/{fileId}`** 로 이동하므로, **알람 UX와 간접적으로 연결**된다.

---

## 14. 초보자를 위한 용어 정리

| 용어 | 이 프로젝트에서의 의미 |
|------|------------------------|
| Redis Pub/Sub | `publish`한 문자열을 구독자가 받는 방식. 여기서는 알람 JSON 브로드캐스트. |
| WebSocket | 브라우저와 서버가 연결을 유지하며 서버가 **먼저** 데이터를 push할 수 있음. |
| Thunk | Redux에서 비동기 API 호출을 처리하는 함수 패턴. |
| `ref_id` / `target_id` | DB `content`에는 `\|` 뒤에 붙고, API/WS JSON에는 `target_id`로 분리되어 온다. |

---

## 15. 파일 경로 빠른 색인

| 구분 | 경로 |
|------|------|
| Redis 루프 | `backend/app/core/alarm_worker.py` |
| WS 허브 | `backend/app/websocket/alarm_manager.py` |
| 알람 생성 | `backend/app/service/alarm_service.py` |
| 알람 API | `backend/app/routers/alarms.py` |
| 공지 API | `backend/app/routers/notice.py` |
| 키워드 구독 API | `backend/app/routers/mypage.py` (`/keywords`) |
| 공지 스키마 | `backend/app/schemas/req/notice.py` |
| 모델 | `backend/app/model/alarms.py`, `notices.py`, `keywords.py` |
| 프론트 알람 HTTP | `frontend/src/components/http/alarmHttp.js` |
| 프론트 공지 HTTP | `frontend/src/components/http/noticeHttp.js` |
| 알람 설정 | `frontend/src/components/ui/alarmConfig.js` |
| 알람 리스트 UI | `frontend/src/components/ui/AlarmList.jsx` |
| WS 훅 | `frontend/src/components/ui/useAlarmSocket.js` |
| 토스트 | `frontend/src/components/ui/ToastBanner.jsx` |
| 사이드바 통합 | `frontend/src/components/ui/SideBar.jsx` |
| 공지 목록 | `frontend/src/components/forum/notice/NoticeList.jsx` |
| 공지 모달 UI | `frontend/src/components/ui/NoticeModal.jsx` |
| 마이페이지 키워드 | `frontend/src/components/forum/mypage/Mypage.jsx` |
| Redux 알람 | `frontend/src/stores/thunks/alarmThunk.js` |
| Redux 공지 | `frontend/src/stores/thunks/noticeThunk.js` |

---

문서 끝. 수정 사항이 생기면 이 파일을 같은 기준으로 갱신하면 된다.
