from colorama import Fore, Back, Style

def log(*args, context=None):
    out = [
        # f"[{Fore.RED}tiki{Fore.YELLOW}trace{Style.RESET_ALL}]"
        f"[{Fore.RED}t{Fore.YELLOW}t{Style.RESET_ALL}]"
    ]

    if context:
        out.append(f"[{Style.DIM}{context}{Style.RESET_ALL}]")

    out += args
    print(*out)