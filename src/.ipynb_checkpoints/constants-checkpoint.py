
import logging

LOG_FN = 'journal.log'

class ActivityLogger():

    def __init__(self, level=logging.INFO, encoding='utf-8'):

        self.level = level
        self.file_handler = logging.FileHandler(LOG_FN, encoding=encoding)
        self.console_handler = logging.StreamHandler()

    def get_logger(self, name):

        file_formatter = logging.Formatter(f'%(asctime)s \nMODULE:{name}, LEVEL:%(levelname)s, LINE:%(lineno)s, MSG:%(message)s\n')
        cons_formatter = logging.Formatter(f'%(asctime)s \n%(lineno)s %(message)s\n\n')

        # Применяем форматтер к обоим Handler
        self.file_handler.setFormatter(file_formatter)
        self.console_handler.setFormatter(cons_formatter)

        # Создаем объект logger и применяем к нему оба Handler
        logger = logging.getLogger(name)
        logger.setLevel(self.level)

        if len(logger.handlers)==0:
            logger.addHandler(self.file_handler)
            logger.addHandler(self.console_handler)

        # Disable propagation of messages to the root logger
        logger.propagate = False

        return logger

    def close_logger(self, logger):

        # import pdb;pdb.set_trace()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        self.file_handler.close()
        self.console_handler.close()