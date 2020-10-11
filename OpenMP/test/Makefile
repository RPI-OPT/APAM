#
# Unittests for MOTAC related functions
# 

# Points to the root of Google Test, relative to where this file is.
# Remember to tweak this if you move this file.
GTEST_DIR = ../lib/googletest

# Where to find test code
USER_DIR = .

SRC_DIR = ../src

INC := -I ../include

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
CPPFLAGS += -isystem $(GTEST_DIR)/include

# Flags passed to the C++ compiler.
CXXFLAGS += -g -Wextra -pthread -fopenmp -lm -lblas -lgfortran -llapack -std=c++0x -w

BLASFLAGS += -lm -lblas

CC := g++
# All tests produced by this Makefile.  Remember to add new tests you
# created to the list.


TESTS = LR_SGD_Momentum.exe LR_SGD_Momentum_par.exe LR_SGD_Momentum_async_par.exe LR_AMSGrad.exe LR_AMSGrad_par.exe LR_AMSGrad_async_par.exe


# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

# House-keeping build targets.

all : $(TESTS)

clean :
	rm -f $(TESTS) gtest.a gtest_main.a *.o

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
gtest-all.o : $(GTEST_SRCS_)
	$(CC) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CC) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# Builds a sample test.  A test should link with either gtest.a or
# gtest_main.a, depending on whether it defines its own main()
# function.

algebra.o : $(SRC_DIR)/algebra.cc
	$(CC) $(CPPFLAGS) $(CXXFLAGS) $(INC) -c $(SRC_DIR)/algebra.cc

nist_spblas.o : $(SRC_DIR)/nist_spblas.cc
	$(CC) $(CPPFLAGS) $(CXXFLAGS) $(INC) -c $(SRC_DIR)/nist_spblas.cc

util.o: $(SRC_DIR)/util.cc
	$(CC) $(CPPFLAGS) $(CXXFLAGS) $(INC) -c $(SRC_DIR)/util.cc

LR_APAM.o : $(USER_DIR)/LR_APAM.cc
	$(CC) $(CPPFLAGS) $(CXXFLAGS) $(INC) -c $(USER_DIR)/LR_APAM.cc

LR_APAM.exe: LR_APAM.o util.o algebra.o nist_spblas.o
	$(CC) $(CPPFLAGS) $(CXXFLAGS) -lpthread -llapack $^ -o $@ $(BLASFLAGS)