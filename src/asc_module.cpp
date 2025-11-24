/*----------------------------------------------------------------------------
| ASC Python Module - Read and decode ASC files with DBC databases
|-----------------------------------------------------------------------------
| This Python C extension module provides functionality to:
| - Read ASC files and extract CAN messages
| - Decode messages using DBC databases
| - Store decoded signals as NumPy arrays organized by message/signal names
|
| Platform: Windows (MinGW GCC/G++)
 ----------------------------------------------------------------------------*/


// Standard library includes
#include <cmath>
#include <cstring>
#include <fstream>

// External library includes
#include "asc_module.h"
#include "canasc/ascparser_opt.h"


// Extract raw value from CAN data bytes
static uint64_t extractRawValue(const uint8_t* data, size_t dataLen, const Vector::DBC::Signal& signal) {
    uint64_t rawValue = 0;

    if (signal.byteOrder == Vector::DBC::ByteOrder::BigEndian) {
        // Motorola (Big Endian) byte order
        uint32_t bitPos = signal.startBit;
        for (uint32_t i = 0; i < signal.bitSize; ++i) {
            uint32_t byteIdx = bitPos / 8;
            uint32_t bitIdx  = 7 - (bitPos % 8);

            if (byteIdx < dataLen) {
                uint64_t bit = (data[byteIdx] >> bitIdx) & 0x01;
                rawValue |= (bit << (signal.bitSize - 1 - i));
            }

            bitPos++;
        }
    } else {
        // Intel (Little Endian) byte order
        uint32_t bitPos = signal.startBit;
        for (uint32_t i = 0; i < signal.bitSize; ++i) {
            uint32_t byteIdx = bitPos / 8;
            uint32_t bitIdx  = bitPos % 8;

            if (byteIdx < dataLen) {
                uint64_t bit = (data[byteIdx] >> bitIdx) & 0x01;
                rawValue |= (bit << i);
            }

            bitPos++;
        }
    }

    // Handle signed values
    if (signal.valueType == Vector::DBC::ValueType::Signed) {
        // Sign-extend if necessary (but only if not full 64-bit)
        // For 64-bit signals, (1ULL << 64) is undefined behavior, and no extension is needed
        if (signal.bitSize < 64) {
            uint64_t signBit = 1ULL << (signal.bitSize - 1);
            if (rawValue & signBit) {
                uint64_t mask = (1ULL << signal.bitSize) - 1;
                rawValue |= ~mask;
            }
        }
    }

    return rawValue;
}

// ASC.__new__
static PyObject* ASC_new(PyTypeObject* type, PyObject* Py_UNUSED(args), PyObject* Py_UNUSED(kwds)) {
    ASCObject* self;
    self = (ASCObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        // Initialize with placement new for C++ objects
        new (&self->messages_data) std::unordered_map<std::string, MessageData>();
        new (&self->dbc_network_cache) std::unordered_map<MessageChannelKey, Vector::DBC::Network*, MessageChannelKeyHash>();
        new (&self->networks) std::vector<Vector::DBC::Network>();
        new (&self->network_channels) std::vector<int>();
        self->initialized = 0;
        self->parsed      = 0;
    }
    return (PyObject*)self;
}

// Helper function to convert Python path object (str or Path) to C string
// Note: The returned string is owned by the Python object and is valid only
// while the path_obj is alive or until the next call to get_path_string.
// Caller should use the string immediately within the same scope as path_obj.
static const char* get_path_string(PyObject* path_obj) {
    // If it's already a string, return it directly
    // The string data is owned by path_obj and remains valid as long as path_obj is alive
    if (PyUnicode_Check(path_obj)) {
        return PyUnicode_AsUTF8(path_obj);
    }

    // If it's a Path object, call __fspath__()
    // We need to keep the returned PyObject alive, so we use a static to cache it
    // This is safe because get_path_string is only called during initialization
    static PyObject* cached_fspath = nullptr;

    if (PyObject_HasAttrString(path_obj, "__fspath__")) {
        // Clear previous cached object if any
        Py_XDECREF(cached_fspath);
        cached_fspath = nullptr;

        PyObject* fspath = PyObject_CallMethod(path_obj, "__fspath__", NULL);
        if (fspath && PyUnicode_Check(fspath)) {
            // Cache the fspath object to keep it alive
            cached_fspath = fspath;
            return PyUnicode_AsUTF8(fspath);
        }
        Py_XDECREF(fspath);
    }

    // Try __str__() as fallback
    // Use the same caching strategy
    Py_XDECREF(cached_fspath);
    cached_fspath = nullptr;

    PyObject* str_obj = PyObject_Str(path_obj);
    if (str_obj && PyUnicode_Check(str_obj)) {
        // Cache the str object to keep it alive
        cached_fspath = str_obj;
        return PyUnicode_AsUTF8(str_obj);
    }
    Py_XDECREF(str_obj);

    return nullptr;
}

// ASC.__init__
static int ASC_init(ASCObject* self, PyObject* args, PyObject* kwds) {
    PyObject* asc_filepath_obj;
    PyObject* channel_dbc_list;

    static char* kwlist[] = {(char*)"asc_filepath", (char*)"channel_dbc_list", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist,
                                     &asc_filepath_obj, &channel_dbc_list)) {
        return -1;
    }

    // Convert ASC filepath (str or Path)
    const char* asc_filepath = get_path_string(asc_filepath_obj);
    if (!asc_filepath) {
        PyErr_SetString(PyExc_TypeError, "asc_filepath must be a string or Path object");
        return -1;
    }

    // Parse list of (channel, dbc_filepath) tuples
    if (!PyList_Check(channel_dbc_list)) {
        PyErr_SetString(PyExc_TypeError, "channel_dbc_list must be a list of (channel, dbc_filepath) tuples");
        return -1;
    }

    Py_ssize_t numEntries = PyList_Size(channel_dbc_list);
    if (numEntries == 0) {
        PyErr_SetString(PyExc_ValueError, "At least one (channel, dbc_filepath) tuple must be provided");
        return -1;
    }

    // Parse each (channel, dbc_filepath) tuple
    for (Py_ssize_t i = 0; i < numEntries; ++i) {
        PyObject* tuple = PyList_GetItem(channel_dbc_list, i);

        if (!PyTuple_Check(tuple) || PyTuple_Size(tuple) != 2) {
            PyErr_SetString(PyExc_TypeError, "Each entry must be a (channel, dbc_filepath) tuple");
            return -1;
        }

        // Extract channel
        PyObject* channel_obj = PyTuple_GetItem(tuple, 0);
        if (!PyLong_Check(channel_obj)) {
            PyErr_SetString(PyExc_TypeError, "Channel must be an integer");
            return -1;
        }
        long channel = PyLong_AsLong(channel_obj);
        if (channel < -1 || channel > 65535) {
            PyErr_SetString(PyExc_ValueError, "Channel must be between -1 (wildcard) and 65535");
            return -1;
        }
        int channel_id = static_cast<int>(channel); // Use int to support -1

        // Extract DBC filepath (str or Path)
        PyObject*   dbc_path_obj = PyTuple_GetItem(tuple, 1);
        const char* dbc_filepath = get_path_string(dbc_path_obj);
        if (!dbc_filepath) {
            PyErr_SetString(PyExc_TypeError, "DBC filepath must be a string or Path object");
            return -1;
        }

        // Load DBC file
        Vector::DBC::Network network;
        std::ifstream        ifs(dbc_filepath);

        if (!ifs.is_open()) {
            PyErr_Format(PyExc_IOError, "Could not open DBC file: %s", dbc_filepath);
            return -1;
        }

        ifs >> network;
        ifs.close();

        if (!network.successfullyParsed) {
            PyErr_Format(PyExc_ValueError, "Failed to parse DBC file: %s", dbc_filepath);
            return -1;
        }

        // Store network and remember which channel it belongs to
        self->networks.push_back(std::move(network));
        self->network_channels.push_back(channel_id); // Track channel for this network
    }

    // Open ASC file using optimized ASCParserOpt
    ASCParserOpt parser(asc_filepath);

    if (parser.fileEnded()) {
        PyErr_Format(PyExc_IOError, "Could not open or parse ASC file: %s", asc_filepath);
        return -1;
    }

    self->initialized = 1;

    // Use temporary structure during collection
    std::unordered_map<std::string, TempMessageData> tempMessagesData;

    // Read all messages from ASC file
    while (!parser.fileEnded()) {
        std::unique_ptr<Message> msg = parser.getMessage();

        if (!msg) {
            continue; // Skip invalid or non-CAN messages
        }

        // Skip error frames and remote frames
        if (msg->is_error_frame() || msg->is_remote_frame()) {
            continue;
        }

        uint32_t       msgId        = static_cast<uint32_t>(msg->arbitration_id());
        uint16_t       msgChannel   = static_cast<uint16_t>(msg->channel());
        const uint8_t* msgData      = msg->data();
        size_t         msgDataSize  = msg->data_size();
        double         timestampSec = msg->timestamp(); // Already in seconds

        // Find matching message in DBC files using cache
        // The cache maps (message_id, channel) -> network index
        const Vector::DBC::Message* dbcMessage = nullptr;

        // Create cache key
        MessageChannelKey cache_key{msgId, msgChannel};

        // Check cache first
        auto cacheIt = self->dbc_network_cache.find(cache_key);
        if (cacheIt != self->dbc_network_cache.end()) {
            // Cache hit - look up message using the cached network pointer
            Vector::DBC::Network* cached_network = cacheIt->second;
            auto                  msgIt          = cached_network->messages.find(msgId);
            if (msgIt != cached_network->messages.end()) {
                dbcMessage = &msgIt->second;
            }
        } else {
            // Cache miss - search all networks and cache the result
            // Check if message channel matches the network's channel
            for (size_t i = 0; i < self->networks.size(); ++i) {
                auto& network         = self->networks[i];
                int   network_channel = self->network_channels[i];

                // Skip if channel doesn't match (unless network channel is wildcard -1)
                if (network_channel != -1 && network_channel != static_cast<int>(msgChannel)) {
                    continue;
                }

                auto msgIt = network.messages.find(msgId);
                if (msgIt != network.messages.end()) {
                    dbcMessage = &msgIt->second;
                    // Cache the network pointer for future messages on this channel
                    self->dbc_network_cache[cache_key] = &network;
                    break;
                }
            }
        }

        if (dbcMessage) {

            auto& msgData_storage = tempMessagesData[dbcMessage->name];

            if (msgData_storage.name.empty()) {
                msgData_storage.name = dbcMessage->name;
                msgData_storage.id   = msgId;
            }

            // Add timestamp to message
            msgData_storage.timestamps.push_back(timestampSec);

            // Decode all signals
            for (const auto& sigPair : dbcMessage->signals) {
                const Vector::DBC::Signal& signal = sigPair.second;

                // Extract raw value
                uint64_t rawValue = extractRawValue(msgData, msgDataSize, signal);

                // Apply scaling immediately during init
                double physicalValue;
                if (signal.valueType == Vector::DBC::ValueType::Signed) {
                    int64_t signedRaw = static_cast<int64_t>(rawValue);
                    physicalValue     = static_cast<double>(signedRaw) * signal.factor + signal.offset;
                } else {
                    physicalValue = static_cast<double>(rawValue) * signal.factor + signal.offset;
                }

                // Store scaled physical value
                auto& sigData = msgData_storage.signals[signal.name];
                if (sigData.name.empty()) {
                    sigData.name = signal.name;

                    // Store metadata on first occurrence (use sanitized name)
                    SignalMetadata& metadata = msgData_storage.signal_metadata[signal.name];
                    metadata.name            = signal.name;
                    metadata.unit            = signal.unit;
                    metadata.factor          = signal.factor;
                    metadata.offset          = signal.offset;
                    metadata.is_signed       = (signal.valueType == Vector::DBC::ValueType::Signed);
                    metadata.bit_size        = signal.bitSize;
                }
                sigData.values.push_back(physicalValue);
            }
        }
    }

    // Convert temporary data to consolidated 2D arrays with scaled values
    for (auto& tempPair : tempMessagesData) {
        const std::string&     msgName  = tempPair.first;
        const TempMessageData& tempMsg  = tempPair.second;
        MessageData&           finalMsg = self->messages_data[msgName];

        finalMsg.name        = tempMsg.name;
        finalMsg.id          = tempMsg.id;
        finalMsg.num_samples = tempMsg.timestamps.size();
        finalMsg.num_signals = tempMsg.signals.size() + 1; // +1 for Time

        // Copy signal metadata
        finalMsg.signal_metadata = tempMsg.signal_metadata;

        // Build ordered signal names list (for consistent column indexing)
        // Column 0 is "Time", followed by other signals
        finalMsg.signal_names.reserve(finalMsg.num_signals);
        finalMsg.signal_names.push_back("Time");
        finalMsg.signal_index_map["Time"] = 0;

        // Add Time signal metadata
        SignalMetadata& timeMeta = finalMsg.signal_metadata["Time"];
        timeMeta.name            = "Time";
        timeMeta.unit            = "s";
        timeMeta.factor          = 1.0;
        timeMeta.offset          = 0.0;
        timeMeta.is_signed       = false;
        timeMeta.bit_size        = 64;

        // Add other signals and build index map
        size_t col_idx = 1;
        for (const auto& sigPair : tempMsg.signals) {
            finalMsg.signal_names.push_back(sigPair.first);
            finalMsg.signal_index_map[sigPair.first] = col_idx++;
        }

        // Allocate consolidated 2D data array: [num_samples * num_signals]
        // Includes Time in column 0
        finalMsg.data.resize(finalMsg.num_samples * finalMsg.num_signals);

        // Fill data row by row (row-major order)
        for (size_t row = 0; row < finalMsg.num_samples; ++row) {
            size_t row_offset = row * finalMsg.num_signals;

            // Column 0: Time
            finalMsg.data[row_offset + 0] = tempMsg.timestamps[row];

            // Columns 1+: Other signals (in order of signal_names)
            for (size_t col = 1; col < finalMsg.num_signals; ++col) {
                const std::string& sigName = finalMsg.signal_names[col];
                auto               sigIt   = tempMsg.signals.find(sigName);
                if (sigIt != tempMsg.signals.end()) {
                    finalMsg.data[row_offset + col] = sigIt->second.values[row];
                } else {
                    finalMsg.data[row_offset + col] = 0.0; // Shouldn't happen
                }
            }
        }
    }

    self->parsed = 1;

    return 0;
}

// ASC.__del__
static void ASC_dealloc(ASCObject* self) {
    if (self->initialized) {
        // Explicitly call destructors for C++ objects
        self->messages_data.~unordered_map();
        self->dbc_network_cache.~unordered_map();
        self->networks.~vector();
        self->network_channels.~vector();
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// ASC.get_signal(message_name, signal_name) -> numpy array
ASC_FASTCALL(ASC_get_signal) {
    ASC_CHECK_PARSED;
    ASC_CHECK_NARGS(2);

    const char* message_name;
    const char* signal_name;
    ASC_GET_STRING_ARG(message_name, 0);
    ASC_GET_STRING_ARG(signal_name, 1);

    ASC_FIND_MESSAGE(msgData, message_name);

    // Find signal column index (works for all signals including "Time")
    int col_idx = msgData.get_signal_index(signal_name);
    if (col_idx < 0) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found in message '%s'",
                     signal_name, message_name);
        return NULL;
    }

    // Create zero-copy strided view of the signal column
    // Data layout: row-major 2D array [num_samples x num_signals]
    // Time is at column 0, other signals at columns 1+
    // We want column col_idx, which has stride = num_signals * sizeof(double)
    npy_intp dims[1]    = {static_cast<npy_intp>(msgData.num_samples)};
    npy_intp strides[1] = {static_cast<npy_intp>(msgData.num_signals * sizeof(double))};

    // Create read-only array (no NPY_ARRAY_WRITEABLE flag) to prevent data corruption
    PyObject* array = PyArray_New(&PyArray_Type, 1, dims, NPY_DOUBLE, strides,
                                  const_cast<double*>(&msgData.data[col_idx]),
                                  sizeof(double), 0, NULL);
    if (array == NULL) {
        return NULL;
    }

    // Set base object to keep ASC object alive
    Py_INCREF(self);
    if (PyArray_SetBaseObject((PyArrayObject*)array, (PyObject*)self) < 0) {
        Py_DECREF(self);
        Py_DECREF(array);
        return NULL;
    }

    return array;
}

// ASC.get_message_names() -> list[str]
ASC_NOARGS(ASC_get_message_names) {
    ASC_CHECK_PARSED;

    PyObject* list;
    ASC_NEW_LIST(list, self->messages_data.size());

    size_t i = 0;
    for (const auto& msgPair : self->messages_data) {
        PyObject* name = sanitized_PyUnicode_FromString(msgPair.first.c_str());
        if (name == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i++, name);
    }

    return list;
}

// ASC.get_signals(message_name) -> list
ASC_FASTCALL(ASC_get_signals) {
    PyObject* list;
    ASC_CHECK_PARSED;
    ASC_CHECK_NARGS(1);

    const char* message_name;
    ASC_GET_STRING_ARG(message_name, 0);

    ASC_FIND_MESSAGE(msgData, message_name);

    // signal_names now includes "Time" at index 0, so just return it directly
    ASC_NEW_LIST(list, msgData.signal_names.size());

    for (size_t i = 0; i < msgData.signal_names.size(); ++i) {
        PyObject* name = sanitized_PyUnicode_FromString(msgData.signal_names[i].c_str());
        if (name == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, name);
    }

    return list;
}

// ASC.get_message_count(message_name) -> int
ASC_FASTCALL(ASC_get_message_count) {
    ASC_CHECK_PARSED;
    ASC_CHECK_NARGS(1);

    const char* message_name;
    ASC_GET_STRING_ARG(message_name, 0);

    ASC_FIND_MESSAGE(msgData, message_name);
    return PyLong_FromSize_t(msgData.num_samples);
}

// ASC.get_signal_units(message_name) -> dict[str, str]
ASC_FASTCALL(ASC_get_signal_units) {
    ASC_CHECK_PARSED;
    ASC_CHECK_NARGS(1);

    const char* message_name;
    ASC_GET_STRING_ARG(message_name, 0);

    ASC_FIND_MESSAGE(msgData, message_name);

    PyObject* dict;
    ASC_NEW_DICT(dict);

    for (const auto& metaPair : msgData.signal_metadata) {
        PyObject* key = sanitized_PyUnicode_FromString(metaPair.first.c_str());
        PyObject* val = sanitized_PyUnicode_FromString(metaPair.second.unit.c_str());

        if (key == NULL || val == NULL) {
            Py_XDECREF(key);
            Py_XDECREF(val);
            Py_DECREF(dict);
            return NULL;
        }

        if (PyDict_SetItem(dict, key, val) < 0) {
            Py_DECREF(key);
            Py_DECREF(val);
            Py_DECREF(dict);
            return NULL;
        }

        Py_DECREF(key);
        Py_DECREF(val);
    }

    return dict;
}

// ASC.get_signal_unit(message_name, signal_name) -> str
ASC_FASTCALL(ASC_get_signal_unit) {
    ASC_CHECK_PARSED;
    ASC_CHECK_NARGS(2);

    const char* message_name;
    const char* signal_name;
    ASC_GET_STRING_ARG(message_name, 0);
    ASC_GET_STRING_ARG(signal_name, 1);

    ASC_FIND_MESSAGE(msgData, message_name);
    ASC_FIND_SIGNAL(msgData, metadata, signal_name);

    return sanitized_PyUnicode_FromString(metadata.unit.c_str());
}

// ASC.get_signal_factors(message_name) -> dict[str, float]
ASC_FASTCALL(ASC_get_signal_factors) {
    ASC_CHECK_PARSED;
    ASC_CHECK_NARGS(1);

    const char* message_name;
    ASC_GET_STRING_ARG(message_name, 0);

    ASC_FIND_MESSAGE(msgData, message_name);

    PyObject* dict;
    ASC_NEW_DICT(dict);

    for (const auto& metaPair : msgData.signal_metadata) {
        PyObject* key = sanitized_PyUnicode_FromString(metaPair.first.c_str());
        PyObject* val = PyFloat_FromDouble(metaPair.second.factor);

        if (key == NULL || val == NULL) {
            Py_XDECREF(key);
            Py_XDECREF(val);
            Py_DECREF(dict);
            return NULL;
        }

        if (PyDict_SetItem(dict, key, val) < 0) {
            Py_DECREF(key);
            Py_DECREF(val);
            Py_DECREF(dict);
            return NULL;
        }

        Py_DECREF(key);
        Py_DECREF(val);
    }

    return dict;
}

// ASC.get_signal_factor(message_name, signal_name) -> float
ASC_FASTCALL(ASC_get_signal_factor) {
    ASC_CHECK_PARSED;
    ASC_CHECK_NARGS(2);

    const char* message_name;
    const char* signal_name;
    ASC_GET_STRING_ARG(message_name, 0);
    ASC_GET_STRING_ARG(signal_name, 1);

    ASC_FIND_MESSAGE(msgData, message_name);
    ASC_FIND_SIGNAL(msgData, metadata, signal_name);

    return PyFloat_FromDouble(metadata.factor);
}

// ASC.get_signal_offsets(message_name) -> dict[str, float]
ASC_FASTCALL(ASC_get_signal_offsets) {
    ASC_CHECK_PARSED;
    ASC_CHECK_NARGS(1);

    const char* message_name;
    ASC_GET_STRING_ARG(message_name, 0);

    ASC_FIND_MESSAGE(msgData, message_name);

    PyObject* dict;
    ASC_NEW_DICT(dict);

    for (const auto& metaPair : msgData.signal_metadata) {
        PyObject* key = sanitized_PyUnicode_FromString(metaPair.first.c_str());
        PyObject* val = PyFloat_FromDouble(metaPair.second.offset);

        if (key == NULL || val == NULL) {
            Py_XDECREF(key);
            Py_XDECREF(val);
            Py_DECREF(dict);
            return NULL;
        }

        if (PyDict_SetItem(dict, key, val) < 0) {
            Py_DECREF(key);
            Py_DECREF(val);
            Py_DECREF(dict);
            return NULL;
        }

        Py_DECREF(key);
        Py_DECREF(val);
    }

    return dict;
}

// ASC.get_signal_offset(message_name, signal_name) -> float
ASC_FASTCALL(ASC_get_signal_offset) {
    ASC_CHECK_PARSED;
    ASC_CHECK_NARGS(2);

    const char* message_name;
    const char* signal_name;
    ASC_GET_STRING_ARG(message_name, 0);
    ASC_GET_STRING_ARG(signal_name, 1);

    ASC_FIND_MESSAGE(msgData, message_name);
    ASC_FIND_SIGNAL(msgData, metadata, signal_name);

    return PyFloat_FromDouble(metadata.offset);
}

// ASC.get_period(message_name) -> int
ASC_FASTCALL(ASC_get_period) {
    ASC_CHECK_PARSED;
    ASC_CHECK_NARGS(1);

    const char* message_name;
    ASC_GET_STRING_ARG(message_name, 0);

    ASC_FIND_MESSAGE(msgData, message_name);

    // Need at least 2 samples to calculate period
    if (msgData.num_samples < 2) {
        PyErr_Format(PyExc_ValueError, "Message '%s' has insufficient samples (%zu) to calculate period",
                     message_name, msgData.num_samples);
        return NULL;
    }

    // Time is always at column 0
    const double* time_data = msgData.data.data(); // Start of flattened 2D array
    size_t        stride    = msgData.num_signals; // Number of columns

    // Get first and last timestamp
    double first_time = time_data[0];                                  // First row, column 0
    double last_time  = time_data[(msgData.num_samples - 1) * stride]; // Last row, column 0

    // Calculate average dt (in seconds)
    double dt = (last_time - first_time) / (msgData.num_samples - 1);

    // Avoid division by zero or negative dt
    if (dt <= 0.0) {
        PyErr_Format(PyExc_ValueError, "Invalid time range for message '%s' (dt = %f)",
                     message_name, dt);
        return NULL;
    }

    // Convert dt to milliseconds and round to nearest integer
    int period_ms = static_cast<int>(std::round(dt * 1000.0));

    return PyLong_FromLong(period_ms);
}

// ASC.get_message_data(message_name) -> 2D numpy array
ASC_FASTCALL(ASC_get_message_data) {
    ASC_CHECK_PARSED;
    ASC_CHECK_NARGS(1);

    const char* message_name;
    ASC_GET_STRING_ARG(message_name, 0);

    ASC_FIND_MESSAGE(msgData, message_name);

    // Return zero-copy 2D view of the data array
    // Data layout: [num_samples x num_signals] where column 0 is Time
    npy_intp dims[2] = {
        static_cast<npy_intp>(msgData.num_samples),
        static_cast<npy_intp>(msgData.num_signals)};

    // Create read-only array (no NPY_ARRAY_WRITEABLE flag) to prevent data corruption
    PyObject* array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, NULL,
                                  const_cast<double*>(msgData.data.data()),
                                  sizeof(double), 0, NULL);
    if (array == NULL) {
        return NULL;
    }

    // Set base object to keep ASC object alive
    Py_INCREF(self);
    if (PyArray_SetBaseObject((PyArrayObject*)array, (PyObject*)self) < 0) {
        Py_DECREF(self);
        Py_DECREF(array);
        return NULL;
    }

    return array;
}

static PyMethodDef ASC_methods[] = {
    { "get_message_names",  (PyCFunction)ASC_get_message_names,   METH_NOARGS,
     "Get list of all message names"                                               },
    {        "get_signal",         (PyCFunction)ASC_get_signal, METH_FASTCALL,
     "Get signal data by message and signal name as numpy array"                   },
    {       "get_signals",        (PyCFunction)ASC_get_signals, METH_FASTCALL,
     "Get list of signal names for a message"                                      },
    { "get_message_count",  (PyCFunction)ASC_get_message_count, METH_FASTCALL,
     "Get number of samples for a message"                                         },
    {  "get_message_data",   (PyCFunction)ASC_get_message_data, METH_FASTCALL,
     "Get entire message as 2D array (time + all signals)"                         },
    {  "get_signal_units",   (PyCFunction)ASC_get_signal_units, METH_FASTCALL,
     "Get all signal units as dictionary"                                          },
    {   "get_signal_unit",    (PyCFunction)ASC_get_signal_unit, METH_FASTCALL,
     "Get unit string for a signal"                                                },
    {"get_signal_factors", (PyCFunction)ASC_get_signal_factors, METH_FASTCALL,
     "Get all signal factors as dictionary"                                        },
    { "get_signal_factor",  (PyCFunction)ASC_get_signal_factor, METH_FASTCALL,
     "Get scaling factor for a signal"                                             },
    {"get_signal_offsets", (PyCFunction)ASC_get_signal_offsets, METH_FASTCALL,
     "Get all signal offsets as dictionary"                                        },
    { "get_signal_offset",  (PyCFunction)ASC_get_signal_offset, METH_FASTCALL,
     "Get scaling offset for a signal"                                             },
    {        "get_period",         (PyCFunction)ASC_get_period, METH_FASTCALL,
     "Get sampling period for a message in milliseconds"                           },
    {                NULL,                                NULL,             0, NULL}
};

// Type definition
static PyTypeObject ASCType = {
    PyVarObject_HEAD_INIT(NULL, 0) "asc_python.ASC", /* tp_name */
    sizeof(ASCObject),                               /* tp_basicsize */
    0,                                               /* tp_itemsize */
    (destructor)ASC_dealloc,                         /* tp_dealloc */
    0,                                               /* tp_vectorcall_offset */
    0,                                               /* tp_getattr */
    0,                                               /* tp_setattr */
    0,                                               /* tp_as_async */
    0,                                               /* tp_repr */
    0,                                               /* tp_as_number */
    0,                                               /* tp_as_sequence */
    0,                                               /* tp_as_mapping */
    0,                                               /* tp_hash */
    0,                                               /* tp_call */
    0,                                               /* tp_str */
    0,                                               /* tp_getattro */
    0,                                               /* tp_setattro */
    0,                                               /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /* tp_flags */
    "ASC file reader with DBC decoding support",     /* tp_doc */
    0,                                               /* tp_traverse */
    0,                                               /* tp_clear */
    0,                                               /* tp_richcompare */
    0,                                               /* tp_weaklistoffset */
    0,                                               /* tp_iter */
    0,                                               /* tp_iternext */
    ASC_methods,                                     /* tp_methods */
    0,                                               /* tp_members */
    0,                                               /* tp_getset */
    0,                                               /* tp_base */
    0,                                               /* tp_dict */
    0,                                               /* tp_descr_get */
    0,                                               /* tp_descr_set */
    0,                                               /* tp_dictoffset */
    (initproc)ASC_init,                              /* tp_init */
    0,                                               /* tp_alloc */
    ASC_new,                                         /* tp_new */
    0,                                               /* tp_free */
    0,                                               /* tp_is_gc */
    0,                                               /* tp_bases */
    0,                                               /* tp_mro */
    0,                                               /* tp_cache */
    0,                                               /* tp_subclasses */
    0,                                               /* tp_weaklist */
    0,                                               /* tp_del */
    0,                                               /* tp_version_tag */
    0,                                               /* tp_finalize */
    0,                                               /* tp_vectorcall */
    0,                                               /* tp_watched */
    0,                                               /* tp_versions_used */
};

// Module definition
static PyModuleDef ascmodule = {
    PyModuleDef_HEAD_INIT,
    "asc_python",                                   /* m_name */
    "ASC file reader and decoder with DBC support", /* m_doc */
    -1,                                             /* m_size */
    NULL,                                           /* m_methods */
    NULL,                                           /* m_slots */
    NULL,                                           /* m_traverse */
    NULL,                                           /* m_clear */
    NULL,                                           /* m_free */
};

// Module initialization
PyMODINIT_FUNC PyInit_asc_python(void) {
    PyObject* m;

    // Initialize NumPy C API
    import_array();

    if (PyType_Ready(&ASCType) < 0)
        return NULL;

    m = PyModule_Create(&ascmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&ASCType);
    if (PyModule_AddObject(m, "ASC", (PyObject*)&ASCType) < 0) {
        Py_DECREF(&ASCType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
