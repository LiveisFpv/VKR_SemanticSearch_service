# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: service.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rservice.proto\x12\x08semantic\"#\n\rSearchRequest\x12\x12\n\nInput_data\x18\x01 \x01(\t\"d\n\rPaperResponse\x12\n\n\x02ID\x18\x01 \x01(\t\x12\r\n\x05Title\x18\x02 \x01(\t\x12\x10\n\x08\x41\x62stract\x18\x03 \x01(\t\x12\x0c\n\x04Year\x18\x04 \x01(\x03\x12\x18\n\x10\x42\x65st_oa_location\x18\x05 \x01(\t\"9\n\x0ePapersResponse\x12\'\n\x06Papers\x18\x01 \x03(\x0b\x32\x17.semantic.PaperResponse\"\xc7\x01\n\nAddRequest\x12\n\n\x02ID\x18\x01 \x01(\t\x12\r\n\x05Title\x18\x02 \x01(\t\x12\x10\n\x08\x41\x62stract\x18\x03 \x01(\t\x12\x0c\n\x04Year\x18\x04 \x01(\x03\x12\x18\n\x10\x42\x65st_oa_location\x18\x05 \x01(\t\x12\x34\n\x10Referenced_works\x18\x06 \x03(\x0b\x32\x1a.semantic.Referenced_works\x12.\n\rRelated_works\x18\x07 \x03(\x0b\x32\x17.semantic.Related_works\"\x1e\n\x10Referenced_works\x12\n\n\x02ID\x18\x01 \x01(\t\"\x1b\n\rRelated_works\x12\n\n\x02ID\x18\x01 \x01(\t\"\x1e\n\rErrorResponse\x12\r\n\x05\x45rror\x18\x01 \x01(\t2\x8e\x01\n\x0fSemanticService\x12@\n\x0bSearchPaper\x12\x17.semantic.SearchRequest\x1a\x18.semantic.PapersResponse\x12\x39\n\x08\x41\x64\x64Paper\x12\x14.semantic.AddRequest\x1a\x17.semantic.ErrorResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_SEARCHREQUEST']._serialized_start=27
  _globals['_SEARCHREQUEST']._serialized_end=62
  _globals['_PAPERRESPONSE']._serialized_start=64
  _globals['_PAPERRESPONSE']._serialized_end=164
  _globals['_PAPERSRESPONSE']._serialized_start=166
  _globals['_PAPERSRESPONSE']._serialized_end=223
  _globals['_ADDREQUEST']._serialized_start=226
  _globals['_ADDREQUEST']._serialized_end=425
  _globals['_REFERENCED_WORKS']._serialized_start=427
  _globals['_REFERENCED_WORKS']._serialized_end=457
  _globals['_RELATED_WORKS']._serialized_start=459
  _globals['_RELATED_WORKS']._serialized_end=486
  _globals['_ERRORRESPONSE']._serialized_start=488
  _globals['_ERRORRESPONSE']._serialized_end=518
  _globals['_SEMANTICSERVICE']._serialized_start=521
  _globals['_SEMANTICSERVICE']._serialized_end=663
# @@protoc_insertion_point(module_scope)
