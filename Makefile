TEST_TARGETS = tests/test_matmul


CXX			=	g++
CXXFLAGS	=	-std=c++17 -Wall -Wextra -Wpedantic -pedantic -Wformat -Wnoexcept -Wcast-qual -fno-exceptions -fno-rtti

UNAME_M	=	$(shell uname -m)

ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64))
	# Use all available CPU extensions, x86 only
	CXXFLAGS	+= 	-march=native -mtune=native
endif

ifdef YAMI_FAST
	CXXFLAGS	+=	-Ofast -flto -DYAMI_FAST
else
	CXXFLAGS	+=	-O0 -g
endif

ifdef YAMI_DEBUG
	CXXFLAGS	+=	-DYAMI_DEBUG
endif

$(info I YAMI build info: )
$(info I Host arch:		$(UNAME_M))
$(info I CXXFLAGS:		$(CXXFLAGS))
$(info I LDFLAGS:		$(LDFLAGS))
$(info I CXX:			$(shell $(CXX) --version | head -n 1))
$(info )

all: clean test

.PHONY: clean test

test: $(TEST_TARGETS)
	@fail=0; \
	total_tests=0; \
	for t in $(TEST_TARGETS); do \
  	  echo "======================================"; \
	  echo "Running $$t"; \
  	  echo "======================================"; \
  	  total_tests=$$((total_tests + 1)); \
  	  ./$$t; \
  	  if [ $$? -ne 0 ]; then \
  	    echo "Test $$t failed!"; \
  	    fail=$$((fail + 1)); \
  	  fi; \
	  echo "======================================"; \
	done; \
	if [ $${fail} -gt 0 ]; then \
	  echo "Failed $$fail/$$total_tests tests!"; \
  	else \
	  echo "All tests passed!"; \
	fi;

clean:
	rm -rf *.o $(TEST_TARGETS) mlp

yami.o: yami.cpp yami.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

mlp: mlp.cpp yami.o
	$(CXX) $(CXXFLAGS) $< -o $@ yami.o $(LDFLAGS)

tests/test_matmul: tests/test_matmul.cpp yami.o
	$(CXX) $(CXXFLAGS) $< -o $@ yami.o $(LDFLAGS)
