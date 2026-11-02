from django import template

register = template.Library()

@register.filter(name='dict_get')
def dict_get(dictionary, key):
    if isinstance(dictionary, dict):
        return dictionary.get(key, '')
    return ''

@register.filter(name='title_format')
def title_format(value):
    if not value:
        return ''
    return str(value).replace('_', ' ').title()
