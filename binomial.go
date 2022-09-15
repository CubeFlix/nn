// binomial.go
// Binomial distribution implementation.

// Adapted From: https://github.com/gonum/gonum/blob/master/stat/distuv/binomial.go
// Copyright ©2018 The Gonum Authors. All rights reserved.
// This code is liscensed under the BSD 3-Clause "New" or "Revised" License.
// Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// The names of its contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package nn

import (
	"math"
	"time"
	"golang.org/x/exp/rand"
)


// Binomial distribution struct.
type binomial struct {
	// N is the total number of Bernoulli trials. N must be greater than 0.
	N float64
	// P is the probability of success in any given trial. P must be in [0, 1].
	P float64

	Src rand.Source
}

// Create the new source.
func (b binomial) NewSource() {
	b.Src = rand.NewSource(uint64(time.Now().UnixNano()))
}

// Rand returns a random sample drawn from the distribution.
func (b binomial) Rand() float64 {
	// NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING (ISBN 0-521-43108-5)
	// p. 295-6
	// http://www.aip.de/groups/soe/local/numres/bookcpdf/c7-3.pdf

	runif := rand.Float64
	rexp := rand.ExpFloat64
	if b.Src != nil {
		rnd := rand.New(b.Src)
		runif = rnd.Float64
		rexp = rnd.ExpFloat64
	}

	p := b.P
	if p > 0.5 {
		p = 1 - p
	}
	am := b.N * p

	if b.N < 25 {
		// Use direct method.
		bnl := 0.0
		for i := 0; i < int(b.N); i++ {
			if runif() < p {
				bnl++
			}
		}
		if p != b.P {
			return b.N - bnl
		}
		return bnl
	}

	if am < 1 {
		// Use rejection method with Poisson proposal.
		const logM = 2.6e-2 // constant for rejection sampling (https://en.wikipedia.org/wiki/Rejection_sampling)
		var bnl float64
		z := -p
		pclog := (1 + 0.5*z) * z / (1 + (1+1.0/6*z)*z) // Padé approximant of log(1 + x)
		for {
			bnl = 0.0
			t := 0.0
			for i := 0; i < int(b.N); i++ {
				t += rexp()
				if t >= am {
					break
				}
				bnl++
			}
			bnlc := b.N - bnl
			z = -bnl / b.N
			log1p := (1 + 0.5*z) * z / (1 + (1+1.0/6*z)*z)
			t = (bnlc+0.5)*log1p + bnl - bnlc*pclog + 1/(12*bnlc) - am + logM // Uses Stirling's expansion of log(n!)
			if rexp() >= t {
				break
			}
		}
		if p != b.P {
			return b.N - bnl
		}
		return bnl
	}
	// Original algorithm samples from a Poisson distribution with the
	// appropriate expected value. However, the Poisson approximation is
	// asymptotic such that the absolute deviation in probability is O(1/n).
	// Rejection sampling produces exact variates with at worst less than 3%
	// rejection with miminal additional computation.

	// Use rejection method with Cauchy proposal.
	g, _ := math.Lgamma(b.N + 1)
	plog := math.Log(p)
	pclog := math.Log1p(-p)
	sq := math.Sqrt(2 * am * (1 - p))
	for {
		var em, y float64
		for {
			y = math.Tan(math.Pi * runif())
			em = sq*y + am
			if em >= 0 && em < b.N+1 {
				break
			}
		}
		em = math.Floor(em)
		lg1, _ := math.Lgamma(em + 1)
		lg2, _ := math.Lgamma(b.N - em + 1)
		t := 1.2 * sq * (1 + y*y) * math.Exp(g-lg1-lg2+em*plog+(b.N-em)*pclog)
		if runif() <= t {
			if p != b.P {
				return b.N - em
			}
			return em
		}
	}
}
