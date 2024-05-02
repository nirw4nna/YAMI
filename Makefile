#TESTS_TARGETS = tests/test_quantize
MODELS = gpt2 llama

CXX			=	g++
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
	CXXFLAGS	+= -DYAMI_FAST -Ofast -ffp-contract=fast -funroll-loops -flto=auto -fuse-linker-plugin
else
	CXXFLAGS	+= -DYAMI_DEBUG -O0 -g
endif

UNAME_S	=	$(shell uname -s)

$(info YAMI build info: )
$(info   OS:		$(UNAME_S))
$(info   ARCH:		$(UNAME_M))
$(info   CXXFLAGS:	$(CXXFLAGS))
$(info   LDFLAGS:	$(LDFLAGS))
$(info   CXX:		$(shell $(CXX) --version | head -n 1))
$(info )

.PHONY: clean pyyami $(MODELS)

clean:
	rm -rf *.o *.so $(MODELS) $(TESTS_TARGETS)

pyyami: src/yami.cpp include/yami.h
	$(CXX) $(CXXFLAGS) -fPIC -shared $< -o yami.so

#test: $(TESTS_TARGETS)
#	@fail=0; \
#	total_tests=0; \
#	for t in $(TESTS_TARGETS); do \
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

llama: models/llama.cpp yami.o yami_utils.o
	$(CXX) $(CXXFLAGS) $< -o $@ yami.o yami_utils.o $(LDFLAGS)
