# Agent Control Status Follow-Ups

Current implementation in UI:
- `AGENT CONTROL` tab now shows `ACTIVE: x/5` in the tab label.
- Status dot in the tab uses this priority:
- `error` agent exists -> red dot
- else `waiting` agent exists -> amber dot
- else at least one `running` agent -> green dot
- else neutral gray dot

What is still mocked/assumed:
- `waiting` is currently treated as "approval required" because no dedicated approval field exists in `Agent`.
- The `5` in `ACTIVE: x/5` is a fixed UI slot count, not backend-configured capacity.

Recommended next additions:
- Add explicit approval fields in `Agent`, for example:
- `approvalRequired: boolean`
- `approvalType: 'tool' | 'human' | 'security'`
- `approvalRequestedAt: string`
- Use `approvalRequired` (not generic `waiting`) to drive amber status.
- Add a small tooltip or legend describing status-dot meaning in the tab row.
- Optionally expose separate counts:
- `RUNNING`
- `WAITING APPROVAL`
- `ERROR`
