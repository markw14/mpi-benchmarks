#pragma once

void device_mem_alloc(char *&device_buf, size_t size_to_alloc);
void host_mem_alloc(char *&host_buf, size_t size_to_alloc);
void device_mem_free(char *&device_buf);
void host_mem_free(char *&host_buf);

