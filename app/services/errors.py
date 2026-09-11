

class ChatError(Exception):
    """Raised for deliberate chat failures (blocked query, missing
    conversation, etc.). The router converts this to a 4xx."""
    pass


class ConversationNotFoundError(Exception):
    pass
