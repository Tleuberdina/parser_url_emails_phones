import re
import urllib.parse
import urllib.request
import urllib.robotparser
import urllib.error
import ssl
import html.parser
from typing import Dict, List, Set, Tuple, Deque
from collections import deque
from dataclasses import dataclass
import time
import socket
from urllib.parse import urlparse, urljoin, urlunparse
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler('website_parser.log', encoding='utf-8')  # Запись в файл
    ]
)

logger = logging.getLogger('RussianWebsiteParser')


@dataclass
class ParserResult:
    """Результат парсинга."""
    url: str
    emails: List[str]
    phones: List[str]
    
    def to_dict(self) -> Dict:
        """Преобразование в требуемый формат."""
        return {
            "url": self.url,
            "emails": sorted(list(set(self.emails))),
            "phones": sorted(list(set(self.phones)))
        }


class RussianContactParser(html.parser.HTMLParser):
    """Парсер для поиска российских контактов."""
    
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.links: Set[str] = set()
        self.emails: Set[str] = set()
        self.phones: Set[str] = set()
        
        # Паттерн для email
        self.email_pattern = re.compile(
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        )
    
    def handle_starttag(self, tag: str, attrs: list):
        attrs_dict = dict(attrs)
        
        # Обработка ссылок
        if tag == 'a' and 'href' in attrs_dict:
            href = attrs_dict['href'].strip()
            if href:
                self._process_href(href)
    
    def handle_data(self, data: str):
        """Обработка текстовых данных."""
        if data.strip():
            self._extract_from_text(data)
    
    def _process_href(self, href: str):
        """Обработка href атрибута."""
        # Email ссылки
        if href.startswith('mailto:'):
            email = href[7:].split('?')[0].strip()
            if self._is_valid_email(email):
                self.emails.add(email.lower())
            return
        
        # Телефонные ссылки
        if href.startswith('tel:'):
            phone = href[4:].strip()
            clean_phone = self._clean_russian_phone(phone)
            if clean_phone:
                self.phones.add(clean_phone)
            return
        
        # Игнорируем специальные ссылки
        if href.startswith(('javascript:', '#', 'skype:')):
            return
        
        # Добавляем обычные ссылки
        absolute_url = urljoin(self.base_url, href)
        self.links.add(absolute_url)
    
    def _extract_from_text(self, text: str):
        """Извлечение данных из текста."""
        # Ищем email
        emails = self.email_pattern.findall(text)
        for email in emails:
            if self._is_valid_email(email):
                self.emails.add(email.lower())
        
        # Ищем телефоны с более строгими паттернами
        phones = self._find_russian_phones(text)
        for phone in phones:
            clean_phone = self._clean_russian_phone(phone)
            if clean_phone:
                self.phones.add(clean_phone)
    
    def _find_russian_phones(self, text: str) -> List[str]:
        """Поиск российских телефонов с контекстной проверкой."""
        phones = []
        
        # Основные форматы российских телефонов
        patterns = [
            # +7 (XXX) XXX-XX-XX
            r'\+7\s?\(?\d{3}\)?\s?\d{3}[- ]?\d{2}[- ]?\d{2}',
            # 8 (XXX) XXX-XX-XX  
            r'8\s?\(?\d{3}\)?\s?\d{3}[- ]?\d{2}[- ]?\d{2}',
            # (XXX) XXX-XX-XX
            r'\(?\d{3}\)?\s?\d{3}[- ]?\d{2}[- ]?\d{2}',
            # XXX-XX-XX (городские с кодом города)
            r'\b\d{3}[- ]?\d{2}[- ]?\d{2}\b',
            # Без кода города: XXX-XX-XX (7 цифр с разделителями)
            r'\b\d{3}[- ]?\d{2}[- ]?\d{2}\b',
            # Без кода города: XX-XX-XX (6 цифр с разделителями)
            r'\b\d{2}[- ]?\d{2}[- ]?\d{2}\b',
            # +7 XXX XXX-XX-XX (без скобок)
            r'\+7\s?\d{3}\s?\d{3}[- ]?\d{2}[- ]?\d{2}',
            # 8-XXX-XXX-XX-XX
            r'8[- ]?\d{3}[- ]?\d{3}[- ]?\d{2}[- ]?\d{2}',
            # (495) 123-45-67
            r'\(?\d{3,4}\)?\s?\d{2,3}[- ]?\d{2}[- ]?\d{2}',
        ]
        
        for pattern in patterns:
            found = re.findall(pattern, text)
            phones.extend(found)
        
        # Дополнительная фильтрация по контексту
        filtered_phones = []
        for phone in phones:
            # Ищем контекст вокруг телефона
            phone_index = text.find(phone)
            if phone_index != -1:
                # Берем текст вокруг телефона (50 символов до и после)
                start = max(0, phone_index - 50)
                end = min(len(text), phone_index + len(phone) + 50)
                context = text[start:end].lower()
                
                # Ключевые слова, указывающие что это телефон
                phone_keywords = [
                    'тел', 'телефон', 'звонит', 'позвони', 'номер',
                    'phone', 'call', 'contact', 'контакт', 'связь',
                    'факс', 'fax', 'моб', 'сотовый', 'мобильный'
                ]
                
                # Знаки, которые часто идут перед/после телефона
                phone_signs = [':', '-', '—', '–', '(', '[', '{']
                
                # Проверяем контекст
                is_likely_phone = False
                
                # 1. Проверка по ключевым словам в контексте
                if any(keyword in context for keyword in phone_keywords):
                    is_likely_phone = True
                
                # 2. Проверка что перед телефоном есть знак препинания или начало строки
                if phone_index > 0:
                    char_before = text[phone_index - 1]
                    if char_before in phone_signs or char_before.isspace():
                        is_likely_phone = True
                
                # 3. Для коротких номеров (6-7 цифр) нужна дополнительная проверка
                digits = re.sub(r'[^\d]', '', phone)
                if 6 <= len(digits) <= 7:
                    # Городские номера часто идут с кодом города или в контексте телефона
                    if '(' in phone or 'тел' in context:
                        is_likely_phone = True
                    else:
                        # Игнорируем просто последовательности цифр без контекста
                        continue
                
                if is_likely_phone:
                    filtered_phones.append(phone)
        
        return filtered_phones
    
    def _is_valid_email(self, email: str) -> bool:
        """Проверка валидности email."""
        email = email.strip()
        
        # Базовая проверка
        if not email or ' ' in email or email.count('@') != 1:
            return False
        
        local, domain = email.split('@')
        
        if not local or not domain or '.' not in domain:
            return False
        
        # Проверка на минимальную длину домена
        if len(domain.split('.')[-1]) < 2:
            return False
        
        # Проверка на специальные символы
        if local.startswith('.') or local.endswith('.') or '..' in local:
            return False
        
        return True
    
    def _clean_russian_phone(self, phone: str) -> str:
        """Очистка и валидация российского номера телефона."""
        # Сохраняем исходную для отладки
        original = phone
        
        # Удаляем все нецифровые символы кроме +
        digits = re.sub(r'[^\d+]', '', phone)
        
        if not digits:
            return ""
        
        # Нормализация: 8 -> +7
        if digits.startswith('8'):
            digits = '7' + digits[1:]
        
        # Добавляем + если его нет и номер начинается с 7
        if digits.startswith('7') and not digits.startswith('+7'):
            digits = '+' + digits
        
        digits_only = digits.replace('+', '')
        
        # Валидация в зависимости от типа номера
        if digits.startswith('+7'):
            # Российские мобильные/городские с кодом страны
            if len(digits_only) != 11:
                return ""  # Должно быть 11 цифр
            
            # Проверка кода оператора (вторая цифра)
            operator_code = digits_only[1]
            
            # Допустимые коды для российских номеров:
            # 9 - мобильные
            # 4 - мобильные (Мегафон, Tele2)
            # 8 - мобильные (Билайн) и специальные (800)
            # 3 - коды городов
            # 4 - коды городов
            # 8 - коды городов и специальные
            
            if operator_code == '9':
                # Мобильные номера
                # Проверка что номер не состоит из повторяющихся цифр
                if len(set(digits_only)) < 3:
                    return ""  # Слишком много одинаковых цифр
            elif operator_code == '8':
                # Может быть мобильный или 8-800
                if digits_only.startswith('7800'):
                    # Бесплатный номер 8-800
                    pass
                else:
                    # Мобильный или городской
                    pass
            elif operator_code in ['3', '4']:
                # Городские коды
                # Проверяем что это не просто случайные цифры
                if digits_only.count('0') > 5:
                    return ""
            else:
                # Другие коды маловероятны для реальных номеров
                return ""
            
            # Проверка на "красивые" номера (часто используются в примерах)
            if digits_only in ['70000000000', '71111111111', '72222222222', 
                               '73333333333', '74444444444', '75555555555',
                               '76666666666', '77777777777', '78888888888',
                               '79999999999']:
                return ""
            
            return digits
        
        elif 6 <= len(digits_only) <= 8:
            # Городские номера без кода страны
            # Проверка что это не просто последовательность одинаковых цифр
            if len(set(digits_only)) == 1:
                return ""  # Все цифры одинаковые
            
            # Проверка на круглые числа (1000000, 2000000 и т.д.)
            if digits_only.endswith('000000') or digits_only.endswith('00000'):
                return ""
            
            # Проверка на часто встречающиеся не-телефоны
            common_non_phones = {
                '1000000', '2000000', '3000000', '4000000', '5000000',
                '123456', '1234567', '12345678', '111111', '222222',
                '333333', '444444', '555555', '666666', '777777',
                '888888', '999999', '100000', '200000', '300000'
            }
            
            if digits_only in common_non_phones:
                return ""
            
            # Для коротких номеров возвращаем как есть (без +)
            return digits_only
        
        return ""
    
    def close(self):
        super().close()


class RussianWebsiteParser:
    """Парсер для российских сайтов."""
    
    def __init__(
        self,
        start_url: str,
        max_pages: int = 100,
        max_depth: int = 3,
        timeout: int = 10,
        delay: float = 0.5
    ):
        # Нормализация URL
        if not start_url.startswith(('http://', 'https://')):
            start_url = 'https://' + start_url
        
        self.start_url = start_url
        self.base_domain = self._get_domain(start_url)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.timeout = timeout
        self.delay = delay
        
        # Структуры данных
        self.visited_urls: Set[str] = set()
        self.urls_to_visit: Deque[Tuple[str, int]] = deque([(start_url, 0)])
        self.all_emails: Set[str] = set()
        self.all_phones: Set[str] = set()
        
        # Настройки запросов
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
        
        # SSL контекст
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # Проверка robots.txt
        self.robots_allowed = self._check_robots_txt()
    
    def _get_domain(self, url: str) -> str:
        """Извлечение домена."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def _check_robots_txt(self) -> bool:
        """Проверка robots.txt - безопасный."""
        try:
            robots_url = urljoin(self.base_domain, '/robots.txt')
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
        
            # Пытаемся загрузить robots.txt
            req = urllib.request.Request(
                robots_url, 
                headers={'User-Agent': self.user_agent}
            )
        
            try:
                response = urllib.request.urlopen(req, timeout=5)
                content = response.read().decode('utf-8', errors='ignore')
                rp.parse(content.splitlines())
            
                # Проверяем разрешен ли доступ
                return rp.can_fetch(self.user_agent, self.start_url)
            
            except urllib.error.HTTPError as e:
                # Если robots.txt возвращает 404 - файла нет
                if e.code == 404:
                    return True  # Нет файла = нет запретов
                else:
                    # Другие HTTP ошибки (403, 500 и т.д.)
                    logger.warning(f"Предупреждение: robots.txt вернул ошибку {e.code}")
                    return False  # Запрещаем при других ошибках
                
            except (urllib.error.URLError, socket.timeout):
                # Сетевые ошибки или таймаут
                logger.warning(f"Предупреждение: не удалось загрузить robots.txt")
                return False  # Запрещаем при сетевых ошибках
            
        except Exception as e:
            # Любая другая ошибка
            logger.error(f"Ошибка при проверке robots.txt: {e}")
            return False  # При ошибках запрещаем доступ
    
    def _normalize_url(self, url: str) -> str:
        """Нормализация URL."""
        try:
            parsed = urlparse(url)
            
            # Заполнение недостающих частей
            scheme = parsed.scheme or 'https'
            netloc = parsed.netloc or urlparse(self.base_domain).netloc
            path = parsed.path or '/'
            
            if not path.startswith('/'):
                path = '/' + path
            
            # Удаляем query и fragment
            return urlunparse((scheme, netloc, path, '', '', ''))
        except:
            return url
    
    def _is_same_domain(self, url: str) -> bool:
        """Проверка того же домена."""
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return True  # Относительная ссылка
            
            base_parsed = urlparse(self.base_domain)
            
            # Нормализация (без www)
            domain1 = parsed.netloc
            if domain1.startswith('www.'):
                domain1 = domain1[4:]
            
            domain2 = base_parsed.netloc
            if domain2.startswith('www.'):
                domain2 = domain2[4:]
            
            return domain1 == domain2
        except:
            return False
    
    def _fetch_page(self, url: str) -> Tuple[str, str]:
        """Загрузка страницы."""
        try:
            if not self.robots_allowed:
                return "", ""
            
            time.sleep(self.delay)
            
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': self.user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
                }
            )
            
            response = urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context)
            
            if response.status != 200:
                return "", ""
            
            # Проверка типа контента
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                return "", ""
            
            # Чтение контента
            html_content = response.read()
            
            # Определение кодировки
            encoding = 'utf-8'
            if 'charset=' in content_type:
                charset_match = re.search(r'charset=([\w-]+)', content_type)
                if charset_match:
                    encoding = charset_match.group(1)
            
            # Декодирование
            try:
                html = html_content.decode(encoding, errors='ignore')
            except:
                html = html_content.decode('utf-8', errors='ignore')
            
            return url, html
            
        except Exception:
            return "", ""
    
    def _extract_from_page(self, html: str, base_url: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """Извлечение данных со страницы."""
        parser = RussianContactParser(base_url)
        try:
            parser.feed(html)
            parser.close()
            
            # Фильтрация ссылок
            filtered_links = set()
            for link in parser.links:
                normalized = self._normalize_url(link)
                if self._is_same_domain(normalized):
                    filtered_links.add(normalized)
            
            return filtered_links, parser.emails, parser.phones
            
        except Exception:
            return set(), set(), set()
    
    def parse(self) -> ParserResult:
        """Основной метод парсинга (без вывода в консоль)."""
        while self.urls_to_visit and len(self.visited_urls) < self.max_pages:
            url, depth = self.urls_to_visit.popleft()
            
            if url in self.visited_urls or depth > self.max_depth:
                continue
            
            self.visited_urls.add(url)
            
            # Загрузка страницы
            fetched_url, html = self._fetch_page(url)
            
            if not html:
                continue
            
            # Извлечение данных
            links, emails, phones = self._extract_from_page(html, url)
            
            # Сохранение данных
            self.all_emails.update(emails)
            self.all_phones.update(phones)
            
            # Добавление новых ссылок
            if depth < self.max_depth and links:
                for link in links:
                    if (link not in self.visited_urls and 
                        not any(link == u for u, _ in self.urls_to_visit)):
                        self.urls_to_visit.append((link, depth + 1))
        
        return ParserResult(
            url=self.start_url,
            emails=list(self.all_emails),
            phones=list(self.all_phones)
        )


def parse_website(start_url: str, **kwargs) -> Dict:
    """
    Основная функция парсера
    
    Args:
        start_url: URL для парсинга
        **kwargs: дополнительные параметры
        
    Returns:
        Результат в требуемом формате.
    """
    try:
        # Валидация URL
        if not start_url.startswith(('http://', 'https://')):
            start_url = 'https://' + start_url
        
        parsed = urlparse(start_url)
        if not parsed.netloc:
            # Возвращаем результат с пустыми списками
            return {
                "url": start_url,
                "emails": [],
                "phones": []
            }
        
        # Параметры парсера
        params = {
            'max_pages': kwargs.get('max_pages', 50),
            'max_depth': kwargs.get('max_depth', 3),
            'timeout': kwargs.get('timeout', 10),
            'delay': kwargs.get('delay', 0.5)
        }
        
        # Создание и запуск парсера
        parser = RussianWebsiteParser(start_url, **params)
        result = parser.parse()
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"ОШИБКА: {str(e)}")
        # В случае любой ошибки возвращаем результат с пустыми списками
        return {
            "url": start_url,
            "emails": [],
            "phones": []
        }


# Минимальный пример использования
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
        result = parse_website(url)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print('Ошибка: не указан адрес сайта.')
        sys.exit(1)
