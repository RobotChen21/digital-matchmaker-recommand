def format_history(messages: list) -> str:
    """将消息列表格式化为 User/AI 文本"""
    if not messages:
        return "(无历史记录)"

    text = []
    for m in messages:
        # 兼容对象属性或字典键值
        role = getattr(m, 'role', None) or m.get('role')
        content = getattr(m, 'content', None) or m.get('content')

        # 只保留 user 和 ai，删除 assistant
        if role == 'user':
            text.append(f"User: {content}")
        elif role == 'ai':
            text.append(f"AI: {content}")

    return "\n".join(text) if text else "(无有效历史记录)"