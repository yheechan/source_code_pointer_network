import recorder

version = 'version1'
proj_list =[
    'boringssl', 'c-ares',
    'freetype2', 'guetzli',
    'harfbuzz', 'libpng',
    'libssh', 'libxml2',
    'pcre', 'proj4',
    'r32', 'sqlite3',
    'vorbis', 'woff2',
    'wpantund'
]

version = 'version3'
proj_list = [
    'total_aspell', 'total_boringssl', 'total_c-ares', 'total_exiv2',
    'total_freetype2', 'total_grok', 'total_guetzli', 'total_harfbuzz',
    'total_lcms', 'total_libarchive', 'total_libexif', 'total_libhtp',
    'total_libpng', 'total_libsndfile', 'total_libssh', 'total_libxml2',
    'total_ndpi', 'total_openthread', 'total_pcre2', 'total_proj4',
    'total_re2', 'total_sqlite3', 'total_usrsctp', 'total_vorbis',
    'total_woff2', 'total_wpantund', 'total_yara', 'total_zstd'
]

recorder.recordData(proj_list, version)