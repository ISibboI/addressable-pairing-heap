//! An addressable pairing heap implementation for Rust.
//! 
//! Addressable heaps return handles to stored elements that make it possible
//! to query and edit them. For example this allows for the `decrease_key(h: Handle)` method
//! that decreases the key (priority) of the element that is associated with the
//! given handle.
//! 
//! This implementation stores elements within a `Stash` that allocates elements
//! densely within an array.
//!
//! It is possible to use custom types as the underlying `Key` type by implementing
//! the `Key` trait.

/// A handle to access stored elements within an addressable pairing heap.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Handle(usize);

impl Handle {
	#[inline]
	fn undef() -> Self {
		Handle(usize::max_value())
	}

	#[inline]
	fn is_undef(self) -> bool {
		self == Handle::undef()
	}
}

impl From<usize> for Handle {
	fn from(val: usize) -> Handle {
		Handle(val)
	}
}

impl From<Handle> for usize {
	fn from(handle: Handle) -> usize {
		handle.0
	}
}

/// Represents a trait for keys within an addressable pairing heap.
/// 
/// A user can use custom type for the key type by implementing this trait.
/// 
/// This trait is implicitly implemented already for all types that
/// are `Copy`, `PartialOrd` and `Ord`.
pub trait Key: Copy + PartialOrd + Ord {}
impl<T> Key for T where T: Copy + PartialOrd + Ord {}

/// An entry within an addressable pairing heap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry<T, K> where K: Key {
	key : K,
	elem: T
}

impl<T, K> Entry<T, K>
	where K: Key
{
	#[inline]
	fn new(key: K, elem: T) -> Self {
		Entry{
			key : key,
			elem: elem
		}
	}

	/// Returns a reference to the key of this entry.
	pub fn key(&self) -> &K {
		&self.key
	}

	/// Returns a reference to the element of this entry.
	pub fn elem(&self) -> &T {
		&self.elem
	}
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Position{
	/// root node at index
	Root(usize),

	/// child of parent with index
	Child(Handle, usize)
}

impl Position {
	#[inline]
	fn child(parent: Handle, index: usize) -> Self {
		Position::Child(parent, index)
	}

	#[inline]
	fn root(index: usize) -> Self {
		Position::Root(index)
	}

	#[inline]
	fn is_root(self) -> bool {
		match self {
			Position::Root(_) => true,
			_                 => false
		}
	}

	#[inline]
	fn is_child(self) -> bool {
		match self {
			Position::Child(..) => true,
			_                   => false
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Node<T, K>
	where K: Key
{
	pos     : Position,
	entry   : Entry<T, K>,
	children: Vec<Handle>
}

impl<T, K> Node<T, K>
	where K: Key
{
	#[inline]
	fn new_root(at: usize, entry: Entry<T, K>) -> Self {
		Node{
			entry   : entry,
			pos     : Position::root(at),
			children: Vec::new()
		}
	}
}

/// Errors that can be caused while using `PairingHeap`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Error {
	/// Caused when using `decrease_key` method with a `new_key` that is greater than the old one.
	DecreaseKeyOutOfOrder
}

/// Generic `Result` type for `PairingHeap` methods.
pub type Result<T> = ::std::result::Result<T, Error>;

use stash::*;

/// Type alias for `PairingHeap` that has `i64` as default `Key` type.
pub type DefaultPairingHeap<T> = PairingHeap<T, i64>;

/// An addressable pairing heap implementation.
/// 
/// Stores elements with an associated key.
/// The key can be thought of as the priority of the element that is associated to it.
/// 
/// Supports usages like `take_min` that takes the element with the minimum key out of this storage.
/// 
/// Inserting elements into this data structure provides the caller with handles
/// that makes accessing the elements possible - this is called "addressable".
/// Handles are always local to the associated pairing heap instance and thus should not be
/// exchanged throughout various instances of pairing heaps.
/// 
/// An special feature of addressable pairing heaps is the possibility to explicitely
/// decrease the key of an already stored element with the `decrease_key` operation which
/// simply increases the priority of the associated element.
/// 
/// It is possible to use different implementations for `Key` as the key type.
#[derive(Debug, Clone)]
pub struct PairingHeap<T, K>
	where K: Key
{
	/// Handle to the element with the minimum key within the pairing heap.
	min: Handle,
	/// The roots of the ```PairingHeap``` where
	/// the first root within this ```Vec``` always represents the one with the minimum ```key```.
	roots: Vec<Handle>,

	/// In the ```data``` vector all elements are stored.
	/// This indirection to the real data allows for efficient addressable elements via handles.
	data: Stash<Node<T, K>, Handle>
}

impl<T, K> PairingHeap<T, K>
	where K: Key
{
	/// Creates a new instance of a `PairingHeap`.
	#[inline]
	pub fn new() -> Self {
		PairingHeap{
			min  : Handle::undef(),
			roots: Vec::new(),
			data : Stash::default()
		}
	}

	/// Clears the `PairingHeap` by clearing the underlying data structures.
	/// As long the stdlib does not deallocate a vectors memory on clear,
	/// this allows for efficient reuse of the roots vector.
	/// No idea about the stash though.
	pub fn clear(&mut self) {
		self.min = Handle::undef();
		self.roots.clear();
		self.data.clear();
	}

	/// Returns the number of elements stored in this `PairingHeap`.
	#[inline]
	pub fn len(&self) -> usize {
		self.data.len()
	}

	/// Returns true if this `PairingHeap` is empty.
	#[inline]
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Returns a reference to the `Node` that is associated with the given handle.
	/// Note that this won't fail on usage for a correct implementation of `PairingHeap`.
	#[inline]
	fn node(&self, handle: Handle) -> &Node<T, K> {
		unsafe{ self.data.get_unchecked(handle) }
	}

	/// Returns a mutable reference to the `Node` that is associated with the given handle.
	/// Note that this won't fail on usage for a correct implementation of `PairingHeap`.
	#[inline]
	fn node_mut(&mut self, handle: Handle) -> &mut Node<T, K> {
		unsafe{ self.data.get_unchecked_mut(handle) }
	}

	/// Links the given `lower` tree under the given `upper` tree thus making `lower`
	/// a children of `upper`.
	fn link(&mut self, upper: Handle, lower: Handle) {

		debug_assert!(upper != lower, "cannot link to self!");
		debug_assert!(self.node(lower).pos.is_root(), "lower cannot have multiple parents!");

		let idx = self.node(upper).children.len();
		self.node_mut(upper).children.push(lower);
		self.node_mut(lower).pos = Position::child(upper, idx);
		self.insert_root(upper);
	}

	/// Links the element with the lower key over the element with the higher key.
	/// Thus making one the child of the other.
	fn union(&mut self, fst: Handle, snd: Handle) {
		debug_assert!(fst != snd, "cannot union self with itself");

		if self.node(fst).entry.key < self.node(snd).entry.key {
			self.link(fst, snd)
		}
		else {
			self.link(snd, fst)
		}
	}

	/// Pairwise unifies roots in the `PairingHeap` which
	/// effectively decreases the number of roots to half.
	fn pairwise_union(&mut self) {
		let mut roots =
			::std::mem::replace(&mut self.roots, Vec::new())
			.into_iter();
		loop {
			match (roots.next(), roots.next()) {
				(Some(fst), Some(snd)) => self.union(fst, snd),
				(Some(fst), None     ) => self.insert_root(fst),
				_                      => return
			}
		}
	}

	/// Updates the internal pointer to the current minimum element by hinting
	/// to a new possible min element within the heap.
	#[inline]
	fn update_min(&mut self, handle: Handle) {
		if self.min.is_undef() || self.node(handle).entry.key < self.node(self.min).entry.key {
			self.min = handle;
		}
	}

	/// Creates a new root node.
	#[inline]
	fn mk_root_node(&mut self, elem: T, key: K) -> Handle {
		let idx = self.len();
		self.data.put(
				Node::new_root(idx, Entry::new(key, elem)))
	}

	/// Inserts a new root into the `PairingHeap` and checks whether it is the new minimum element.
	fn insert_root(&mut self, new_root: Handle) {
		let idx = self.roots.len();
		self.roots.push(new_root);
		self.node_mut(new_root).pos = Position::root(idx);
		self.update_min(new_root);
	}

	/// Inserts the given element into the `PairingHeap` with its associated key
	/// and returns a `Handle` to it that allows to directly address it.
	/// 
	/// The handle is for example required in order to use methods like `decrease_key`.
	#[inline]
	pub fn push(&mut self, elem: T, key: K) -> Handle {
		let handle = self.mk_root_node(elem, key);
		self.insert_root(handle);
		handle

	}

	/// Cuts the given `child` from its parent and inserts it as a root into the `PairingHeap`.
	/// Will panic if the given `child` is not a child and thus a root node already.
	fn cut(&mut self, child: Handle) {
		debug_assert!(self.node(child).pos.is_child());

		match self.node(child).pos {
			Position::Root(_) => unsafe{ ::unreachable::unreachable() },
			Position::Child(parent, idx) => {
				self.node_mut(parent).children.swap_remove(idx);
				self.node_mut(child).pos = Position::root(self.len());
				self.insert_root(child);
			}
		}
	}

	/// Decreases the key of the element with the associated given `handle`.
	/// Will panic if the given new key is not lower than the previous key.
	pub fn decrease_key(&mut self, handle: Handle, new_key: K) -> Result<()> {
		if new_key >= self.node(handle).entry.key {
			return Err(Error::DecreaseKeyOutOfOrder)
		}

		self.node_mut(handle).entry.key = new_key;
		match self.node(handle).pos {
			Position::Root(_) => {
				self.update_min(handle);
			},
			Position::Child(..) => {
				self.cut(handle)
			}
		}
		Ok(())
	}

	/// Returns a reference to the element associated with the given handle.
	#[inline]
	pub fn get(&self, handle: Handle) -> Option<&Entry<T, K>> {
		self.data
			.get(handle)
			.and_then(|node| Some(&node.entry))
	}

	/// Returns a mutable reference to the element associated with the given handle.
	#[inline]
	pub fn get_mut(&mut self, handle: Handle) -> Option<&mut Entry<T, K>> {
		self.data
			.get_mut(handle)
			.and_then(|node| Some(&mut node.entry))
	}

	/// Returns a reference to the element associated with the given handle.
	/// 
	/// Does not perform bounds checking so use it carefully!
	#[inline]
	pub unsafe fn get_unchecked(&self, handle: Handle) -> &Entry<T, K> {
		&self.node(handle).entry
	}

	/// Returns a mutable reference to the element associated with the given handle.
	/// 
	/// Does not perform bounds checking so use it carefully!
	#[inline]
	pub unsafe fn get_unchecked_mut(&mut self, handle: Handle) -> &mut Entry<T, K> {
		&mut self.node_mut(handle).entry
	}

	/// Returns a reference to the current minimum element if not empty.
	#[inline]
	pub fn peek(&self) -> Option<&Entry<T, K>> {
		self.get(self.min)
	}

	/// Returns a reference to the current minimum element.
	/// 
	/// Does not perform bounds checking so use it carefully!
	#[inline]
	pub unsafe fn peek_unchecked(&self) -> &Entry<T, K> {
		self.get_unchecked(self.min)
	}

	/// Returns a mutable reference to the current minimum element if not empty.
	#[inline]
	pub fn peek_mut(&mut self) -> Option<&mut Entry<T, K>> {
		let min = self.min;
		self.get_mut(min)
	}

	/// Returns a reference to the current minimum element without bounds checking.
	/// So use it very carefully!
	#[inline]
	pub unsafe fn peek_unchecked_mut(&mut self) -> &mut Entry<T, K> {
		let min = self.min;
		self.get_unchecked_mut(min)
	}

	/// Removes the element associated with the minimum key within this `PairingHeap` and returns it.
	#[inline]
	pub fn pop(&mut self) -> Option<Entry<T, K>> {
		match self.is_empty() {
			true => None,
			_    => unsafe{ Some(self.pop_unchecked()) }
		}
	}

	/// Removes the element associated with the minimum key within this `PairingHeap` without
	/// checking for emptiness and returns it.
	/// 
	/// So use this method carefully!
	pub unsafe fn pop_unchecked(&mut self) -> Entry<T, K> {
		let min = self.min;
		match self.node(min).pos {
			Position::Child(..) => ::unreachable::unreachable(),
			Position::Root(idx) => {
				self.roots.swap_remove(idx);
				self.min = Handle::undef();
				for child in ::std::mem::replace(&mut self.node_mut(min).children, Vec::new()).into_iter() {
					self.insert_root(child);
				}
				self.pairwise_union();
				self.data.take_unchecked(min).entry
			}
		}
	}

	/// Iterate over the values in this `PairingHeap` by reference in unspecified order.
	#[inline]
	pub fn values<'a>(&'a self) -> Values<'a, T, K> {
		Values{iter: self.data.values()}
	}

	/// Iterate over the values in this `PairingHeap` by mutable reference unspecified order.
	#[inline]
	pub fn values_mut<'a>(&'a mut self) -> ValuesMut<'a, T, K> {
		ValuesMut{iter: self.data.values_mut()}
	}

	/// Iterate over values stored within a `PairingHeap` in a sorted-by-min order. Drains the heap.
	#[inline]
	pub fn drain_min(self) -> DrainMin<T, K> {
		DrainMin{heap: self}
	}
}

use std::ops::{Index, IndexMut};

impl<T, K> Index<Handle> for PairingHeap<T, K>
	where K: Key
{
	type Output = T;

	fn index(&self, handle: Handle) -> &Self::Output {
		&self.data
			.get(handle)
			.expect("no node found for given handle")
			.entry.elem
	}
}

impl<T, K> IndexMut<Handle> for PairingHeap<T, K>
	where K: Key
{
	fn index_mut(&mut self, handle: Handle) -> &mut Self::Output {
		&mut self.data
			.get_mut(handle)
			.expect("no node found for given handle")
			.entry.elem
	}
}

/// Iterator over references to values stored within a `PairingHeap`.
pub struct Values<'a, T: 'a, K: 'a + Key> {
	iter: ::stash::stash::Values<'a, Node<T, K>>
}

/// Iterator over mutable references to values stored within a `PairingHeap`.
pub struct ValuesMut<'a, T: 'a, K: 'a + Key> {
	iter: ::stash::stash::ValuesMut<'a, Node<T, K>>
}

impl<'a, T, K: Key> Iterator for Values<'a, T, K> {
	type Item = &'a T;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		self.iter.next().map(|node| &node.entry.elem)
	}
}

impl<'a, T, K: Key> Iterator for ValuesMut<'a, T, K> {
	type Item = &'a mut T;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		self.iter.next().map(|node| &mut node.entry.elem)
	}
}

/// Iterator over values stored within a `PairingHeap` in a sorted-by-min order. Drains the heap.
pub struct DrainMin<T, K: Key> {
	heap: PairingHeap<T, K>
}

impl<T, K: Key> Iterator for DrainMin<T, K> {
	type Item = Entry<T, K>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		self.heap.pop()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn take_min() {
		let mut ph = PairingHeap::new();
		ph.push(0,   6);
		ph.push(1,  10);
		ph.push(2, -42);
		ph.push(3,1337);
		ph.push(4,  -1);
		ph.push(5,   1);
		ph.push(6,   2);
		ph.push(7,   3);
		ph.push(8,   4);
		ph.push(9,   5);
		assert_eq!(Some(Entry::new(-42, 2)), ph.pop());
		assert_eq!(Some(Entry::new(-1, 4)), ph.pop());
		assert_eq!(Some(Entry::new(1, 5)), ph.pop());
		assert_eq!(Some(Entry::new(2, 6)), ph.pop());
		assert_eq!(Some(Entry::new(3, 7)), ph.pop());
		assert_eq!(Some(Entry::new(4, 8)), ph.pop());
		assert_eq!(Some(Entry::new(5, 9)), ph.pop());
		assert_eq!(Some(Entry::new(6, 0)), ph.pop());
		assert_eq!(Some(Entry::new(10, 1)), ph.pop());
		assert_eq!(Some(Entry::new(1337, 3)), ph.pop());
		assert_eq!(None   , ph.pop());
	}

	#[test]
	fn decrease_key() {
		let mut ph = PairingHeap::new();
		let a = ph.push(0,   0);
		let b = ph.push(1,  50);
		let c = ph.push(2, 100);
		let d = ph.push(3, 150);
		let e = ph.push(4, 200);
		let f = ph.push(5, 250);
		assert_eq!(Some(&Entry::new(0, 0)), ph.peek());
		assert_eq!(Ok(()), ph.decrease_key(f, -50));
		assert_eq!(Some(&Entry::new(-50, 5)), ph.peek());
		assert_eq!(Ok(()), ph.decrease_key(e, -100));
		assert_eq!(Some(&Entry::new(-100, 4)), ph.peek());
		assert_eq!(Ok(()), ph.decrease_key(d, -99));
		assert_eq!(Some(&Entry::new(-100, 4)), ph.peek());
		assert_eq!(Err(Error::DecreaseKeyOutOfOrder), ph.decrease_key(c, 1000));
		assert_eq!(Some(&Entry::new(-100, 4)), ph.peek());
		assert_eq!(Ok(()), ph.decrease_key(b, -1000));
		assert_eq!(Some(&Entry::new(-1000, 1)), ph.peek());
		assert_eq!(Err(Error::DecreaseKeyOutOfOrder), ph.decrease_key(a, 100));
		assert_eq!(Some(&Entry::new(-1000, 1)), ph.peek());
	}

	#[test]
	fn empty_take() {
		let mut ph = PairingHeap::<usize, usize>::new();
		assert_eq!(None, ph.pop());
	}

	fn setup() -> PairingHeap<char, i64> {
		let mut ph = PairingHeap::new();
		ph.push('a', 100);
		ph.push('b',  50);
		ph.push('c', 150);
		ph.push('d', -25);
		ph.push('e', 999);
		ph.push('f',  42);
		ph.push('g',  43);
		ph.push('i',  41);
		ph.push('j',-100);
		ph.push('k', -77);
		ph.push('l', 123);
		ph.push('m',-123);
		ph.push('n',   0);
		ph.push('o',  -1);
		ph.push('p',   2);
		ph.push('q',  -3);
		ph.push('r',   4);
		ph.push('s',  -5);
		ph
	}

	#[test]
	fn drain_min() {
		let ph = setup();
		let mut drain = ph.drain_min();

		assert_eq!(drain.next(), Some(Entry::new(-123, 'm')));
		assert_eq!(drain.next(), Some(Entry::new(-100, 'j')));
		assert_eq!(drain.next(), Some(Entry::new(-77, 'k')));
		assert_eq!(drain.next(), Some(Entry::new(-25, 'd')));
		assert_eq!(drain.next(), Some(Entry::new(-5, 's')));
		assert_eq!(drain.next(), Some(Entry::new(-3, 'q')));
		assert_eq!(drain.next(), Some(Entry::new(-1, 'o')));
		assert_eq!(drain.next(), Some(Entry::new(0, 'n')));

		assert_eq!(drain.next(), Some(Entry::new(2, 'p')));
		assert_eq!(drain.next(), Some(Entry::new(4, 'r')));
		assert_eq!(drain.next(), Some(Entry::new(41, 'i')));
		assert_eq!(drain.next(), Some(Entry::new(42, 'f')));
		assert_eq!(drain.next(), Some(Entry::new(43, 'g')));
		assert_eq!(drain.next(), Some(Entry::new(50, 'b')));
		assert_eq!(drain.next(), Some(Entry::new(100, 'a')));
		assert_eq!(drain.next(), Some(Entry::new(123, 'l')));
		assert_eq!(drain.next(), Some(Entry::new(150, 'c')));
		assert_eq!(drain.next(), Some(Entry::new(999, 'e')));

		assert_eq!(drain.next(), None);
	}

	#[test]
	fn values() {
		let ph = setup();
		let values = ph.values();

		// cannot test order of values since it is unspecified!
		assert_eq!(values.count(), 18);
	}
}

#[cfg(all(feature = "bench", test))]
mod bench {
	use super::*;
    use test::{Bencher, black_box};
    // use ::std::collections::BinaryHeap;

	fn setup_sample() -> Vec<i64> {
		use rand::{thread_rng, sample};
		let n       = 100_000;
		let mut rng = thread_rng();
		sample(&mut rng, 1..n, n as usize)
	}

	fn setup_sample_bigpod() -> Vec<BigPod> {
		use rand::{thread_rng, sample};
		let n       = 100_000;
		let mut rng = thread_rng();
		sample(&mut rng, 1..n, n as usize)
			.into_iter()
			.map(|val| val.into())
			.collect::<Vec<BigPod>>()
	}

    #[derive(Debug, Clone, PartialEq, Eq, Ord)]
    struct BigPod {
    	elems0: [i64; 32],
    	elems1: [i64; 32],
    	elems2: [i64; 32],
    	elems3: [i64; 32]
    }

    impl From<i64> for BigPod {
    	fn from(val: i64) -> BigPod {
    		let mut bp = BigPod{
    			elems0: [0; 32],
    			elems1: [1; 32],
    			elems2: [2; 32],
    			elems3: [3; 32]
    		};
    		bp.elems0[0] = val;
    		bp
    	}
    }

    impl PartialOrd for BigPod {
    	fn partial_cmp(&self, other: &BigPod) -> Option<::std::cmp::Ordering> {
    		self.elems0[0].partial_cmp(&other.elems0[0])
    	}
    }

	#[bench]
	fn vec_pairing_heap_push(bencher: &mut Bencher) {
		let sample = setup_sample();
		bencher.iter(|| {
			let mut ph = PairingHeap::new();
			for &key in sample.iter() {
				black_box(ph.push((), key));
			}
		});
	}

	#[bench]
	fn vec_pairing_heap_push_bigpod(bencher: &mut Bencher) {
		let sample = setup_sample_bigpod();
		bencher.iter(|| {
			let mut ph = PairingHeap::new();
			for bigpod in sample.iter() {
				black_box(ph.push(bigpod.clone(), bigpod.elems0[0]));
			}
		});
	}

	// #[bench]
	// fn binary_heap_push(bencher: &mut Bencher) {
	// 	let sample = setup_sample();
	// 	bencher.iter(|| {
	// 		let mut bh = BinaryHeap::new();
	// 		for &key in sample.iter() {
	// 			black_box(bh.push(key));
	// 		}
	// 	});
	// }

	// #[bench]
	// fn binary_heap_push_bigpod(bencher: &mut Bencher) {
	// 	let sample = setup_sample_bigpod();
	// 	bencher.iter(|| {
	// 		let mut bh = BinaryHeap::new();
	// 		for bigpod in sample.iter() {
	// 			black_box(bh.push(bigpod.clone()));
	// 		}
	// 	});
	// }

	#[bench]
	fn vec_pairing_heap_pop(bencher: &mut Bencher) {
		let mut ph = PairingHeap::new();
		for key in setup_sample().into_iter() {
			ph.push((), key);
		}
		bencher.iter(|| {
			let mut ph = ph.clone();
			while let Some(_) = black_box(ph.pop()) {}
		});
	}

	#[bench]
	fn vec_pairing_heap_pop_bigpod(bencher: &mut Bencher) {
		let mut ph = PairingHeap::new();
		for bigpod in setup_sample_bigpod().into_iter() {
			let head = bigpod.elems0[0];
			ph.push(bigpod, head);
		}
		bencher.iter(|| {
			let mut ph = ph.clone();
			while let Some(_) = black_box(ph.pop()) {}
		});
	}

	// #[bench]
	// fn binary_heap_pop(bencher: &mut Bencher) {
	// 	let mut bh = BinaryHeap::new();
	// 	for key in setup_sample().into_iter() {
	// 		bh.push(key);
	// 	}
	// 	bencher.iter(|| {
	// 		let mut bh = bh.clone();
	// 		while let Some(_) = black_box(bh.pop()) {}
	// 	});
	// }

	// #[bench]
	// fn binary_heap_pop_bigpod(bencher: &mut Bencher) {
	// 	let mut bh = BinaryHeap::new();
	// 	for bigpod in setup_sample_bigpod().into_iter() {
	// 		bh.push(bigpod);
	// 	}
	// 	bencher.iter(|| {
	// 		let mut bh = bh.clone();
	// 		while let Some(_) = black_box(bh.pop()) {}
	// 	});
	// }

	#[bench]
	fn vec_pairing_heap_clone(bencher: &mut Bencher) {
		let mut ph = PairingHeap::new();
		for key in setup_sample().into_iter() {
			ph.push((), key);
		}
		bencher.iter(|| {
			black_box(&ph.clone());
		});
	}

	// #[bench]
	// fn binary_heap_clone(bencher: &mut Bencher) {
	// 	let mut bh = BinaryHeap::new();
	// 	for key in setup_sample().into_iter() {
	// 		bh.push(key);
	// 	}
	// 	bencher.iter(|| {
	// 		black_box(&bh.clone());
	// 	});
	// }
}
