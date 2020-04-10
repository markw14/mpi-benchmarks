#pragma once

enum transfer_t { MAIN, WORKLOAD };

size_t device_get_num_of_dev();
void device_set_current(size_t n);
void device_set_current(const std::string &pci_id);
char *device_alloc_mem(size_t size);
void device_free_mem(char *ptr);
bool device_is_idle();
void device_submit_workload(int ncycles);
void d2h_transfer(char *to, char *from, size_t size, transfer_t type = transfer_t::MAIN);
void h2d_transfer(char *to, char *from, size_t size, transfer_t type = transfer_t::MAIN);
