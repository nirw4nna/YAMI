TEST_TARGETS = test/test_matmul

CXX			=	g++
# -fno-align-loops -fno-align-labels are interesting options, they should provide (at least the second one) some benefits
CXXFLAGS	=	-std=c++17 -Wall -Wextra -Wshadow -Wformat -Wnoexcept -Wcast-qual -Wunused -Wdouble-promotion \
 				-Wlogical-op -Wcast-align -fno-exceptions -fno-rtti -pthread
LDFLAGS		=	-lm

UNAME_M	=	$(shell uname -m)

ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64))
	# Use all available CPU extensions, x86 only
	CXXFLAGS	+= 	-march=native -mtune=native
endif

ifdef YAMI_FAST
	CXXFLAGS	+= -Ofast -flto=auto -fuse-linker-plugin -DYAMI_FAST
else
	CXXFLAGS	+= -O0 -g -DYAMI_DEBUG
endif

UNAME_S	=	$(shell uname -s)

$(info YAMI build info: )
$(info   OS:		$(UNAME_S))
$(info   ARCH:		$(UNAME_M))
$(info   CXXFLAGS:	$(CXXFLAGS))
$(info   LDFLAGS:	$(LDFLAGS))
$(info   CXX:		$(shell $(CXX) --version | head -n 1))
$(info )

all: clean pyyami gpt2

.PHONY: clean pyyami gpt2

clean:
	rm -rf *.o *.so $(TEST_TARGETS) mlp gpt2 main

pyyami: yami.cpp
	$(CXX) $(CXXFLAGS) -fPIC -shared $< -o yami.so

test: $(TEST_TARGETS)
	@#fail=0; \
#	total_tests=0; \
#	for t in $(TEST_TARGETS); do \
#  	  echo "======================================"; \
#	  echo "Running $$t"; \
#  	  echo "======================================"; \
#  	  total_tests=$$((total_tests + 1)); \
#  	  ./$$t; \
#  	  if [ $$? -ne 0 ]; then \
#  	    echo "Test $$t failed!"; \
#  	    fail=$$((fail + 1)); \
#  	  fi; \
#	  echo "======================================"; \
#	done; \
#	if [ $${fail} -gt 0 ]; then \
#	  echo "Failed $$fail/$$total_tests tests!"; \
#  	else \
#	  echo "All tests passed!"; \
#	fi;

yami.o: yami.cpp yami.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

yami_utils.o: yami_utils.cpp yami_utils.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

gpt2: gpt2.cpp yami.o yami_utils.o
	$(CXX) $(CXXFLAGS) $< -o $@ yami.o yami_utils.o $(LDFLAGS)

test/test_matmul: test/test_matmul.cpp yami.o
	$(CXX) $(CXXFLAGS) $< -o $@ yami.o $(LDFLAGS)
