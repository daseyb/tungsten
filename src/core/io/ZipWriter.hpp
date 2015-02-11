#ifndef ZIPWRITER_HPP_
#define ZIPWRITER_HPP_

#include <miniz/miniz.h>

namespace Tungsten {

class Path;

class ZipWriter
{
    mz_zip_archive _archive;
    bool _open;

public:
    ZipWriter(const Path &dst);
    ~ZipWriter();

    bool addFile(const Path &src, const Path &dst, int compressionLevel = 5);
    bool addFile(const void *src, size_t len, const Path &dst, int compressionLevel = 5);
    bool addDirectory(const Path &dst);

    void close();
};

}

#endif /* ZIPWRITER_HPP_ */
