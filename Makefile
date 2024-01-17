#TESTS_TARGETS = tests/test_matmul

CXX			=	g++
# -fno-align-loops -fno-align-labels are interesting options, they should provide (at least the second one) some benefits
# also -fprefetch-loop-arrays
CXXFLAGS	=	-std=c++17 -I./include/ -Wall -Wextra -Wshadow -Wformat -Wnoexcept -Wcast-qual -Wunused -Wdouble-promotion \
 				-Wlogical-op -Wcast-align -fno-exceptions -fno-rtti -pthread
LDFLAGS		=	-lm

UNAME_M	=	$(shell uname -m)

ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64))
	# Use all available CPU extensions, x86 only
	CXXFLAGS	+= 	-march=native -mtune=native
endif

# At some point I should introduce "levels", for example logging each time a tensor is created could be enabled
# only at the highest debug level.
ifdef YAMI_FAST
	CXXFLAGS	+= -Ofast -g -flto=auto -fuse-linker-plugin -DYAMI_FAST
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


.PHONY: clean pyyami gpt2

clean:
	rm -rf *.o *.so gpt2

pyyami: src/yami.cpp include/yami.h
	$(CXX) $(CXXFLAGS) -fPIC -shared $< -o yami.so

#test: $(TESTS_TARGETS)
#	@#fail=0; \
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

yami.o: src/yami.cpp include/yami.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

yami_utils.o: src/yami_utils.cpp include/yami_utils.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

gpt2: models/gpt2.cpp yami.o yami_utils.o
	$(CXX) $(CXXFLAGS) $< -o $@ yami.o yami_utils.o $(LDFLAGS)

#tests/test_matmul: tests/test_matmul.cpp yami.o
#	$(CXX) $(CXXFLAGS) $< -o $@ yami.o $(LDFLAGS)
