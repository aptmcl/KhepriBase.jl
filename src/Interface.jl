using Base: @kwdef
using Reexport
@reexport using KhepriBase                # user-facing API (KhepriBase's `export` set)
KhepriBase.@import_backend_api            # developer API (KhepriBase's `public` set) into backend scope
using Dates
using ColorTypes

# resolve name clashes with other deps
using KhepriBase:
  XYZ,
  Text

import Base:
  show
