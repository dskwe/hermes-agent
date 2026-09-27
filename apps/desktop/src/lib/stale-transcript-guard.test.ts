import { describe, expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import type { ChatMessage } from './chat-messages'
import { textPart, toChatMessages } from './chat-messages'
import { messagesIfTranscriptBehind } from './stale-transcript-guard'

/**
 * The guard refuses a send when the authoritative latest page holds MORE
 * transcript than the window does, rather than forking the session (#65047).
 * It measures that difference with `remoteChat.length > localMessages.length`
 * after `toChatMessages`.
 *
 * A backend-authored NOTICE also arrives as a `ChatMessage` (`model changed`,
 * `background agent work finished`): `tui_gateway/server.py` persists
 * `display_kind=model_switch` with `role=user` on an in-place model switch, and
 * hydration renders it as a system row. Nothing about it means another window
 * exists — but it inflates the page by one message per event, so the guard
 * reported "This window was behind another view of the same chat" to a user who
 * had only switched models, refused the send, and said the same thing on every
 * retry. Staleness must be measured in AUTHORED content, not array length.
 *
 * A retained (head-trimmed) store also holds FEWER messages than the page
 * without anything being behind (#123909): transcript retention (#77311)
 * releases the head of the stored transcript, so a long tool-heavy chat renders
 * its latest 120-row page to more messages than the store keeps. There the
 * durable tip row decides: same last persisted row on both sides = current.
 */

const row = (over: Partial<SessionMessage> & Pick<SessionMessage, 'role'>): SessionMessage => ({
  content: '',
  timestamp: 1_700_000_000,
  ...over
})

const userTurn = (id: number, text: string): SessionMessage =>
  row({ content: text, id, role: 'user', timestamp: 1_700_000_000 + id })

const assistantTurn = (id: number, text: string): SessionMessage =>
  row({ content: text, id, role: 'assistant', timestamp: 1_700_000_000 + id })

/** What an in-place model switch persists, exactly as the gateway writes it. */
const modelSwitchNotice = (id: number): SessionMessage =>
  row({
    content: 'switched to another model',
    display_kind: 'model_switch',
    id,
    role: 'user',
    timestamp: 1_700_000_000 + id
  })

/**
 * A store row the retention pass kept: rendered content without a durable
 * `rowId`, because it was hydrated from an older page the store no longer
 * holds the backend rows of.
 */
const retainedRow = (id: string, role: ChatMessage['role'], text: string): ChatMessage =>
  ({
    id,
    parts: [textPart(text)],
    role
  }) as ChatMessage

describe('messagesIfTranscriptBehind', () => {
  it('does not treat a backend-authored notice as another view being ahead', () => {
    const windowMessages = toChatMessages([userTurn(1, 'ask'), assistantTurn(2, 'answer')])
    const page = toChatMessages([userTurn(1, 'ask'), assistantTurn(2, 'answer'), modelSwitchNotice(3)])

    // The notice is a real message in the page: this is what skews a length compare.
    expect(page).toHaveLength(3)
    expect(page.map(message => message.role)).toEqual(['user', 'assistant', 'system'])

    expect(messagesIfTranscriptBehind(windowMessages, page)).toBeNull()
  })

  it('still refuses when another view advanced the chat with a real turn', () => {
    const windowMessages = toChatMessages([userTurn(1, 'ask'), assistantTurn(2, 'answer')])

    const page = toChatMessages([
      userTurn(1, 'ask'),
      assistantTurn(2, 'answer'),
      userTurn(4, 'sent from another window')
    ])

    expect(messagesIfTranscriptBehind(windowMessages, page)).not.toBeNull()
  })

  it('still refuses when a notice arrives alongside a reply this window never saw', () => {
    const windowMessages = toChatMessages([userTurn(1, 'ask'), assistantTurn(2, 'answer')])

    const page = toChatMessages([
      userTurn(1, 'ask'),
      assistantTurn(2, 'answer'),
      modelSwitchNotice(3),
      assistantTurn(4, 'a reply this window never rendered')
    ])

    expect(messagesIfTranscriptBehind(windowMessages, page)).not.toBeNull()
  })

  it('is current when both sides hold the same notices and the same turns', () => {
    const rows = [userTurn(1, 'ask'), assistantTurn(2, 'answer'), modelSwitchNotice(3)]

    expect(messagesIfTranscriptBehind(toChatMessages(rows), toChatMessages(rows))).toBeNull()
  })

  it('is current when the page is empty, and refreshes a window that holds nothing', () => {
    const rows = [userTurn(1, 'ask'), assistantTurn(2, 'answer')]

    expect(messagesIfTranscriptBehind(toChatMessages(rows), [])).toBeNull()
    expect(messagesIfTranscriptBehind([], toChatMessages(rows))).toEqual(toChatMessages(rows))
  })

  it('does not read retention head-trimming as another window being ahead (#123909)', () => {
    // A long tool-heavy chat: the store legitimately holds fewer messages than
    // the latest page renders to, because retention released the head. The
    // durable tip is the same row on both sides — nothing is behind.
    const stored: ChatMessage[] = [
      retainedRow('old-1', 'user', 'morning ask'),
      retainedRow('old-2', 'assistant', 'morning answer'),
      ...toChatMessages([userTurn(9, 'latest ask'), assistantTurn(10, 'latest answer')])
    ]
    const page = toChatMessages([
      userTurn(1, 'morning ask'),
      assistantTurn(2, 'morning answer'),
      userTurn(9, 'latest ask'),
      assistantTurn(10, 'latest answer')
    ])

    // The length compare this regression pins: the page renders to more.
    expect(stored.length).toBeLessThan(page.length)

    expect(messagesIfTranscriptBehind(stored, page)).toBeNull()
  })

  it('still refuses when another window advanced the chat past the shared tip', () => {
    const stored = toChatMessages([userTurn(1, 'ask'), assistantTurn(2, 'answer')])
    const page = toChatMessages([
      userTurn(1, 'ask'),
      assistantTurn(2, 'answer'),
      userTurn(3, 'sent from another window')
    ])

    expect(messagesIfTranscriptBehind(stored, page)).not.toBeNull()
  })

  it('still counts when the stored transcript has no durable row to tip on', () => {
    const stored: ChatMessage[] = [
      retainedRow('old-1', 'user', 'old ask'),
      retainedRow('old-2', 'assistant', 'old answer')
    ]
    const page = toChatMessages([
      userTurn(1, 'old ask'),
      assistantTurn(2, 'old answer'),
      userTurn(3, 'never seen here')
    ])

    expect(messagesIfTranscriptBehind(stored, page)).not.toBeNull()
  })
})
