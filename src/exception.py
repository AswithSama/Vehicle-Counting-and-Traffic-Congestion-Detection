import sys
from logger import logging
import logging
def get_error_details(error, error_info: sys):
    _, _, traceback = error_info.exc_info()
    script_name = traceback.tb_frame.f_code.co_filename
    error_message = "An error occurred in script: [{0}] at line: [{1}] with message: [{2}]".format(
        script_name, traceback.tb_lineno, str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_info: sys):
        super().__init__(error_message)
        self.error_message = get_error_details(error_message, error_info=error_info)
    
    def __str__(self):
        return self.error_message
    

