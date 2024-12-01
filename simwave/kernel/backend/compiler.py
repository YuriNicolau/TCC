import os
from hashlib import sha1


class Compiler:
    """
    Base class to implement the runtime compiler.

    Parameters
    ----------
    cc : str, optional
        C compiler. Default is gcc.    
    cflags : str, optional
        C compiler flags.
    """
    def __init__(self, cc, cflags):
        self.cc = cc        
        self.cflags = cflags        

    def compile(self, float_precision, program_path):
        """
        Compile the program.

        Parameters
        ----------       
        float_precision : str
            Float single (C float) or double (C double) precision.
        program_path : str
            Path to the C file
        Returns
        ----------
        str
            Path to the compiled shared object
        """
        # get the working dir
        working_dir = os.getcwd()        

        # get c file content
        with open(program_path, 'r', encoding='utf-8') as f:
            c_file_content = f.read()

        # object root dir
        object_dir = working_dir + "/tmp/"        

        # compose the object string
        object_str = "{} {} {}\n{}".format(            
            self.cc,
            self.cflags,
            float_precision,
            c_file_content
        )

        # apply sha1 hash to name the object
        hash = sha1()
        hash.update(object_str.encode())
        object_name = hash.hexdigest() + ".so"

        # object complete path
        object_path = object_dir + object_name

        # check if object_file already exists
        if os.path.exists(object_path):
            print("Shared object already compiled in:", object_path)
        else:
            cmd = (
                self.cc
                + " "
                + program_path
                + " "
                + self.cflags
                + " {}".format(float_precision)                
                + " -o "
                + object_path
            )

            print("Compilation command:", cmd)

            # create a dir to save the compiled shared object
            os.makedirs(object_dir, exist_ok=True)

            # execute the command
            if os.system(cmd) != 0:
                raise Exception("Compilation failed")

        return object_path
