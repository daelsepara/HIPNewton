#ifndef __HIP_CHECK_H__
#define __HIP_CHECK_H__

#define HIP_CHECK(status)                                                                                                                 \
    {                                                                                                                                     \
        if (status != hipSuccess)                                                                                                         \
        {                                                                                                                                 \
            std::cerr << std::endl                                                                                                        \
                      << "error: '" << hipGetErrorString(status) << "'(" << status << ") at " << __FILE__ << ":" << __LINE__ << std::endl \
                      << std::endl;                                                                                                       \
            exit(0);                                                                                                                      \
        }                                                                                                                                 \
    }

#endif
