/* MIT, fscikit-learn contributors.
 * Validation-only workaround for GCC libfortran PR128183 on MinGW/UCRT.
 * See https://gcc.gnu.org/PR128183 and
 * https://github.com/jmarrec/libgfortran-ucrt-setlocale .
 * No GCC source is included here. The Fortran library does not link this file.
 * Link single-threaded validation executables with -static-libgfortran and
 * -Wl,--wrap=setlocale. Keep the queried numeric-locale name alive across the
 * runtime's temporary switch to C, instead of returning UCRT's borrowed name.
 */
#include <locale.h>
#include <stdlib.h>
#include <string.h>
char *__real_setlocale(int category, const char *name);
char *__wrap_setlocale(int category, const char *name) {
    static char numeric_name[4096];
    char *result = __real_setlocale(category, name);
    if (category == LC_NUMERIC && name == NULL && result != NULL) {
        size_t count = strlen(result) + 1;
        if (count > sizeof numeric_name) abort();
        memcpy(numeric_name, result, count);
        return numeric_name;
    }
    return result;
}
